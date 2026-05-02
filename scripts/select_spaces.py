"""
Parking Space ROI Selection Tool.
Interactive OpenCV window that lets users define parking slots as 4-point
polygons on a reference image. Polygons are saved to a pickle file with
resolution metadata for use during detection.

Usage:
    python scripts/select_spaces.py
    python scripts/select_spaces.py --image lot_B.png --output lot_B_ROIs.pkl
    python scripts/select_spaces.py --help

Controls:
    Left-click : Add polygon corner point (4 points = 1 slot)
    Right-click: Remove an existing slot
    Q          : Save and quit
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on the path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROI_PICKLE, REF_IMAGE

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive parking slot polygon annotation tool.",
    )
    parser.add_argument(
        "--image", type=str, default=REF_IMAGE,
        help=f"Path to the reference image / screenshot of the parking lot (default: {REF_IMAGE})",
    )
    parser.add_argument(
        "--output", type=str, default=ROI_PICKLE,
        help=f"Path to save the polygon ROI pickle file (default: {ROI_PICKLE})",
    )
    return parser.parse_args()


args = parse_args()
ref_image_path = args.image
roi_output_path = args.output

# --- Load existing polygons (supports both old and new formats) ---
try:
    with open(roi_output_path, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            posList = data.get("polygons", [])
        else:
            posList = data
    logger.info("Loaded %d saved parking slots from %s", len(posList), roi_output_path)
except FileNotFoundError:
    posList = []
    logger.warning("No existing slots found at %s. Starting fresh.", roi_output_path)

polygon_points: list[tuple[int, int]] = []

# --- Display setup ---
img0 = cv2.imread(ref_image_path)
if img0 is None:
    raise FileNotFoundError(f"Could not load {ref_image_path}. Make sure it exists in the folder.")

h0, w0 = img0.shape[:2]

# Scale down large images to fit screen (max 1280px wide)
MAX_DISPLAY_WIDTH = 1280
if w0 > MAX_DISPLAY_WIDTH:
    display_scale = MAX_DISPLAY_WIDTH / w0
    display_w = MAX_DISPLAY_WIDTH
    display_h = int(h0 * display_scale)
    logger.info("Image too large (%dx%d). Scaling to %dx%d for display.", w0, h0, display_w, display_h)
else:
    display_scale = 1.0
    display_w, display_h = w0, h0


def to_original(x: int, y: int) -> tuple[int, int]:
    """Convert display coordinates to original image coordinates."""
    return int(x / display_scale), int(y / display_scale)


def to_display(x: int, y: int) -> tuple[int, int]:
    """Convert original coordinates to display coordinates."""
    return int(x * display_scale), int(y * display_scale)


def mouseClick(event: int, x: int, y: int, flags: int, params) -> None:
    """Handle mouse events for polygon annotation."""
    global polygon_points, posList

    # Convert display click to original image coordinates
    ox, oy = to_original(x, y)

    # Left-click: mark polygon corners (stored in original coordinates)
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((ox, oy))
        logger.info("Point added: (%d, %d)", ox, oy)
        if len(polygon_points) == 4:
            posList.append(polygon_points.copy())
            save()
            logger.info("Added slot #%d", len(posList))
            polygon_points.clear()

    # Right-click: remove a slot (check in original coordinates)
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, polygon in enumerate(posList):
            if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (ox, oy), False) >= 0:
                posList.pop(i)
                save()
                logger.info("Removed slot #%d", i + 1)
                break


def save() -> None:
    """Save polygons and reference image size to pickle."""
    with open(roi_output_path, "wb") as f:
        pickle.dump({"size": (w0, h0), "polygons": posList}, f)
    logger.info("Saved %d slots to %s", len(posList), roi_output_path)


cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
logger.info("Image: %s (%dx%d) | Output: %s", ref_image_path, w0, h0, roi_output_path)
logger.info("Left-click 4 points per slot | Right-click inside a slot to delete | Press 'Q' to save & quit")

while True:
    img = img0.copy()

    # Draw saved polygons (in original coordinates)
    for idx, polygon in enumerate(posList, start=1):
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        cx = int(np.mean([p[0] for p in polygon]))
        cy = int(np.mean([p[1] for p in polygon]))
        cv2.putText(img, str(idx), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show in-progress points
    for point in polygon_points:
        cv2.circle(img, point, 5, (0, 255, 0), -1)

    # Resize for display
    if display_scale != 1.0:
        img = cv2.resize(img, (display_w, display_h))

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        save()
        break

cv2.destroyAllWindows()
logger.info("Polygon marking session ended. All data saved.")