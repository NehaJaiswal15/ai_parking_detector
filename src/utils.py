"""
Utility functions for the AI Parking Space Detection System.
Provides YOLO detection wrappers, polygon drawing, label rendering,
and resolution-aware polygon scaling.
"""

import logging
import cv2
import numpy as np
from ultralytics import YOLO

from src.config import (
    CONFIDENCE_THRESHOLD, VEHICLE_CLASSES, YOLO_IMGSZ,
    COLOR_OCCUPIED, COLOR_AVAILABLE, OVERLAY_ALPHA, COLOR_LABEL_BG,
)

logger = logging.getLogger(__name__)


def scale_polygons(
    polys: list[list[tuple[int, int]]],
    ref_size: tuple[int, int],
    cur_size: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    """Scale polygon ROIs from reference resolution to current frame resolution.

    Args:
        polys: List of polygons, each polygon is a list of (x, y) points.
        ref_size: (width, height) of the reference image used during annotation.
        cur_size: (height, width) of the current video frame.

    Returns:
        List of scaled polygons matching the current frame resolution.
    """
    ref_w, ref_h = ref_size
    cur_h, cur_w = cur_size
    sx, sy = cur_w / ref_w, cur_h / ref_h
    return [[(int(x * sx), int(y * sy)) for (x, y) in poly] for poly in polys]


def YOLO_Detection(
    model: YOLO,
    frame: np.ndarray,
    conf: float = CONFIDENCE_THRESHOLD,
) -> tuple[list, list, list, dict]:
    """Run YOLO inference on a single frame and return detections.

    Filters results to vehicle classes only (car, bus, truck).

    Args:
        model: Loaded Ultralytics YOLO model instance.
        frame: BGR image as a NumPy array.
        conf: Minimum confidence threshold for detections.

    Returns:
        Tuple of (boxes, classes, confidences, class_names):
        - boxes: List of [x1, y1, x2, y2] bounding boxes.
        - classes: List of class IDs for each detection.
        - confidences: List of confidence scores for each detection.
        - class_names: Dict mapping class ID to human-readable name.
    """
    results = model.predict(frame, conf=conf, classes=VEHICLE_CLASSES, imgsz=YOLO_IMGSZ)
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    confs = results[0].boxes.conf.tolist()
    names = results[0].names

    logger.debug("Detected %d vehicles (conf >= %.2f)", len(boxes), conf)
    return boxes, classes, confs, names


def label_detection(
    frame: np.ndarray,
    text: str,
    x1: float, y1: float, x2: float, y2: float,
    tbox_color: tuple[int, int, int] = COLOR_LABEL_BG,
    fontFace: int = 1,
    fontScale: float = 0.8,
    fontThickness: int = 1,
) -> None:
    """Draw a bounding box with a text label on the frame.

    Args:
        frame: BGR image (modified in-place).
        text: Label string to display above the box.
        x1, y1: Top-left corner of the bounding box.
        x2, y2: Bottom-right corner of the bounding box.
        tbox_color: BGR color for the bounding box and label background.
        fontFace: OpenCV font face constant.
        fontScale: Font size scaling factor.
        fontThickness: Thickness of the rendered text.
    """
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), tbox_color, 2)
    (text_w, text_h), _ = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    y_adjust = 10
    cv2.rectangle(frame, (int(x1), int(y1) - text_h - y_adjust),
                  (int(x1) + text_w + y_adjust, int(y1)), tbox_color, -1)
    cv2.putText(frame, text, (int(x1) + 5, int(y1) - 5),
                fontFace, fontScale, (255, 255, 255), fontThickness, cv2.LINE_AA)


def drawPolygons(
    frame: np.ndarray,
    points_list: list[list[tuple[int, int]]],
    detection_points: list[tuple[int, int]] | None = None,
    polygon_color_inside: tuple[int, int, int] = COLOR_OCCUPIED,
    polygon_color_outside: tuple[int, int, int] = COLOR_AVAILABLE,
    alpha: float = OVERLAY_ALPHA,
) -> tuple[np.ndarray, int, list[bool]]:
    """Draw parking slot polygons with occupancy status on the frame.

    Each polygon is color-coded (red = occupied, green = available) and
    blended onto the frame with transparency. Slot numbers and status
    labels are rendered at each polygon's centroid.

    Args:
        frame: BGR image to draw on.
        points_list: List of polygons defining parking slots.
        detection_points: List of (x, y) centers of detected vehicles.
        polygon_color_inside: BGR color for occupied slots.
        polygon_color_outside: BGR color for available slots.
        alpha: Transparency factor for the polygon overlay (0.0–1.0).

    Returns:
        Tuple of (annotated_frame, occupied_count, slot_status):
        - annotated_frame: Frame with polygons drawn.
        - occupied_count: Number of occupied slots.
        - slot_status: List of booleans (True = occupied) per slot.
    """
    overlay = frame.copy()
    occupied_polygons = 0
    slot_status: list[bool] = []

    for idx, area in enumerate(points_list, start=1):
        area_np = np.array(area, np.int32)

        # Check whether any detection center falls inside this polygon
        is_inside = any(
            cv2.pointPolygonTest(area_np, pt, False) >= 0
            for pt in (detection_points or [])
        )

        color = polygon_color_inside if is_inside else polygon_color_outside
        if is_inside:
            occupied_polygons += 1
        slot_status.append(is_inside)

        # Draw filled polygon
        cv2.fillPoly(overlay, [area_np], color)

        # Compute centroid for label placement
        cx = int(np.mean([p[0] for p in area]))
        cy = int(np.mean([p[1] for p in area]))

        label = "OCCUPIED" if is_inside else "AVAILABLE"
        text_color = (255, 255, 255) if is_inside else (0, 0, 0)

        # Draw slot number badge
        cv2.circle(overlay, (cx, cy - 25), 12, (255, 255, 255), -1)
        cv2.putText(overlay, str(idx), (cx - 8, cy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw status text
        cv2.putText(overlay, label, (cx - 45, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    # Blend overlay with original frame
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    logger.debug("Slots: %d occupied / %d total", occupied_polygons, len(points_list))
    return frame, occupied_polygons, slot_status