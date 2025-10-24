import cv2
import numpy as np

def YOLO_Detection(model, frame, conf=0.35):
    # Detect only vehicle classes (COCO: car=2, bus=5, truck=7)
    results = model.predict(frame, conf=conf, classes=[2, 5, 7])
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    return boxes, classes, names


def label_detection(frame, text, left, top, bottom, right,
                    tbox_color=(30, 155, 50), fontFace=1, fontScale=0.8, fontThickness=1):
    cv2.rectangle(frame, (int(left), int(top)), (int(bottom), int(right)), tbox_color, 2)
    (text_w, text_h), _ = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    y_adjust = 10
    cv2.rectangle(frame, (int(left), int(top) - text_h - y_adjust),
                  (int(left) + text_w + y_adjust, int(top)), tbox_color, -1)
    cv2.putText(frame, text, (int(left) + 5, int(top) - 5),
                fontFace, fontScale, (255, 255, 255), fontThickness, cv2.LINE_AA)


def drawPolygons(frame, points_list, detection_points=None,
                 polygon_color_inside=(0, 0, 255),   # red for occupied
                 polygon_color_outside=(0, 255, 0),  # green for available
                 alpha=0.4):
    """
    Draws each parking slot polygon and labels it.
    Returns the frame, count of occupied slots, and list of boolean occupancy.
    """
    overlay = frame.copy()
    occupied_polygons = 0
    slot_status = []  # list of True/False for each slot

    for idx, area in enumerate(points_list, start=1):
        area_np = np.array(area, np.int32)

        # check whether any detection point falls inside polygon
        is_inside = any(
            cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in (detection_points or [])
        )

        color = polygon_color_inside if is_inside else polygon_color_outside
        if is_inside:
            occupied_polygons += 1
        slot_status.append(is_inside)

        # draw filled polygon
        cv2.fillPoly(overlay, [area_np], color)

        # compute center for label
        cx = int(np.mean([p[0] for p in area]))
        cy = int(np.mean([p[1] for p in area]))

        label = "OCCUPIED" if is_inside else "AVAILABLE"
        text_color = (255, 255, 255) if is_inside else (0, 0, 0)

        # draw slot number
        cv2.circle(overlay, (cx, cy - 25), 12, (255, 255, 255), -1)
        cv2.putText(overlay, str(idx), (cx - 8, cy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # draw status text
        cv2.putText(overlay, label, (cx - 45, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    # blend overlay
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame, occupied_polygons, slot_status