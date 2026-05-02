"""
AI Parking Space Detection — CLI Entry Point.
Runs real-time parking occupancy detection on a video file
using YOLOv11 and displays results in an OpenCV window.

Usage:
    python main.py
    python main.py --video path/to/video.mp4 --model yolo11n.pt --conf 0.4
    python main.py --help
"""

import argparse
import logging
import time

import cv2
import numpy as np

from src.config import MODEL_PATH, ROI_PICKLE, VIDEO_PATH, CONFIDENCE_THRESHOLD, DISPLAY_MAX_WIDTH
from src.detector import ParkingDetector
from src.occupancy_logger import OccupancyLogger
from src.utils import label_detection, scale_polygons

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Parking Space Detection System — Real-time occupancy analysis.",
    )
    parser.add_argument(
        "--video", type=str, default=VIDEO_PATH,
        help=f"Path to the parking-lot video file (default: {VIDEO_PATH})",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH,
        help=f"Path to the YOLO model weights (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--roi", type=str, default=ROI_PICKLE,
        help=f"Path to the ROI pickle file (default: {ROI_PICKLE})",
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Detection confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Disable CSV occupancy logging",
    )
    return parser.parse_args()


def main() -> None:
    """Run the parking detection pipeline with an OpenCV display window."""
    args = parse_args()

    logger.info("Starting AI Parking Detection")
    logger.info("Video: %s | Model: %s | Confidence: %.2f", args.video, args.model, args.conf)

    # Initialize detector
    detector = ParkingDetector(
        model_path=args.model,
        roi_path=args.roi,
        confidence=args.conf,
    )

    # Initialize CSV logger
    csv_logger = None if args.no_csv else OccupancyLogger()

    # FPS tracking
    fps = 0.0
    prev_time = time.time()

    try:
        for frame_num, total_frames, result in detector.process_video(args.video):

            # --- Calculate FPS ---
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
            prev_time = current_time

            frame = result.frame

            # --- Log to CSV ---
            if csv_logger:
                csv_logger.log(
                    frame_num=frame_num,
                    total_slots=result.total_slots,
                    occupied=result.occupied,
                    available=result.available,
                    occupancy_pct=result.occupancy_percent,
                )

            # --- Top-right Parking Status Panel ---
            panel_x, panel_y = frame.shape[1] - 300, 10
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 290, panel_y + 180), (40, 40, 40), -1)
            cv2.putText(frame, "PARKING STATUS", (panel_x + 10, panel_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Total: {result.total_slots}", (panel_x + 10, panel_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Occupied: {result.occupied}", (panel_x + 10, panel_y + 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Available: {result.available}", (panel_x + 10, panel_y + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Occupancy: {result.occupancy_percent:.1f}%", (panel_x + 10, panel_y + 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 10, panel_y + 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- Progress bar ---
            bar_x0, bar_y0, bar_w, bar_h = panel_x + 150, panel_y + 145, 120, 15
            cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (100, 100, 100), 1)
            fill = int(bar_w * result.occupancy_percent / 100)
            bar_color = (0, 0, 255) if result.occupancy_percent > 50 else (0, 255, 0)
            cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + fill, bar_y0 + bar_h), bar_color, -1)
            cv2.putText(frame, f"Frame: {frame_num}/{total_frames}",
                        (panel_x + 80, panel_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # --- Draw vehicle bounding boxes ---
            polys = detector.roi_polygons
            ref = detector.ref_size if detector.ref_size else (frame.shape[1], frame.shape[0])
            scaled_polys = scale_polygons(polys, ref, frame.shape[:2])

            for (x1, y1, x2, y2), cls in zip(result.boxes, result.classes):
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                in_poly = any(
                    cv2.pointPolygonTest(np.array(p, np.int32), center, False) >= 0
                    for p in scaled_polys
                )
                name = result.names[int(cls)]
                color = (0, 0, 255) if in_poly else (0, 255, 0)
                label_detection(frame, text=str(name), tbox_color=color, x1=x1, y1=y1, x2=x2, y2=y2)

            # --- Resize for display if frame is too wide ---
            display_frame = frame
            h_disp, w_disp = frame.shape[:2]
            if w_disp > DISPLAY_MAX_WIDTH:
                scale = DISPLAY_MAX_WIDTH / w_disp
                display_frame = cv2.resize(frame, (int(w_disp * scale), int(h_disp * scale)))

            cv2.imshow("AI Parking Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User requested quit at frame %d", frame_num)
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        logger.info("Detection session ended.")


if __name__ == "__main__":
    main()