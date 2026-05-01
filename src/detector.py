"""
Core detection engine for the AI Parking Space Detection System.
Wraps YOLO model loading, ROI management, and per-frame occupancy
detection into a single reusable class.
"""

import logging
import pickle
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import (
    MODEL_PATH, ROI_PICKLE, CONFIDENCE_THRESHOLD,
    VEHICLE_CLASSES, OVERLAY_ALPHA,
)
from src.utils import scale_polygons, YOLO_Detection, drawPolygons, label_detection

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Structured output from a single frame detection."""

    frame: np.ndarray
    total_slots: int
    occupied: int
    available: int
    occupancy_percent: float
    slot_status: list[bool] = field(default_factory=list)
    boxes: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    confidences: list = field(default_factory=list)
    names: dict = field(default_factory=dict)


class ParkingDetector:
    """End-to-end parking occupancy detector.

    Loads a YOLO model and pre-defined parking slot polygons, then
    provides methods to run detection on individual frames or stream
    results from a video file.

    Args:
        model_path: Path to the YOLO weights file.
        roi_path: Path to the pickle file containing polygon ROIs.
        confidence: Minimum confidence threshold for vehicle detection.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        roi_path: str = ROI_PICKLE,
        confidence: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.confidence = confidence

        # --- Device selection (GPU if available, else CPU) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # --- Load YOLO model ---
        self.model = YOLO(model_path)
        self.model.to(self.device)
        logger.info("Loaded model: %s", model_path)

        # --- Load ROI polygons ---
        self.roi_polygons, self.ref_size = self._load_rois(roi_path)
        logger.info("Loaded %d parking slot ROIs", len(self.roi_polygons))

    @staticmethod
    def _load_rois(roi_path: str) -> tuple[list, tuple | None]:
        """Load parking slot polygons from a pickle file.

        Supports both the new dict format (with resolution metadata)
        and the legacy list-only format.

        Args:
            roi_path: Path to the ROI pickle file.

        Returns:
            Tuple of (polygons, reference_size). reference_size is None
            if the legacy format is detected.
        """
        with open(roi_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            return data["polygons"], data["size"]
        return data, None

    def detect_frame(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a single frame and return structured results.

        Args:
            frame: BGR image as a NumPy array.

        Returns:
            DetectionResult with annotated frame, occupancy stats, and
            per-slot status.
        """
        cur_h, cur_w = frame.shape[:2]

        # Resolve reference size on first frame if using legacy ROI format
        ref = self.ref_size if self.ref_size else (cur_w, cur_h)

        # Scale polygons to match current frame resolution
        polys = scale_polygons(self.roi_polygons, ref, (cur_h, cur_w))

        # Run YOLO detection
        boxes, classes, confs, names = YOLO_Detection(
            self.model, frame, conf=self.confidence
        )

        # Compute detection centers for point-in-polygon test
        detection_points = [
            (int((x1 + x2) / 2), int((y1 + y2) / 2))
            for (x1, y1, x2, y2) in boxes
        ]

        # Draw polygon overlays and get occupancy status
        frame, occupied, slot_status = drawPolygons(
            frame, polys, detection_points=detection_points
        )

        total = len(polys)
        available = total - occupied
        occ_pct = (occupied / total) * 100 if total > 0 else 0.0

        return DetectionResult(
            frame=frame,
            total_slots=total,
            occupied=occupied,
            available=available,
            occupancy_percent=occ_pct,
            slot_status=slot_status,
            boxes=boxes,
            classes=classes,
            confidences=confs,
            names=names,
        )

    def process_video(self, video_path: str):
        """Yield DetectionResult for each frame of a video.

        Args:
            video_path: Path to the input video file.

        Yields:
            Tuple of (frame_number, total_frames, DetectionResult).
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_num = 0

        logger.info("Processing video: %s (%d frames)", video_path, total_frames)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1
                result = self.detect_frame(frame)
                yield frame_num, total_frames, result
        finally:
            cap.release()
            logger.info("Video processing complete. %d frames processed.", frame_num)
