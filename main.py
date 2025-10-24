import cv2, torch, numpy as np, pickle
from ultralytics import YOLO
from utilis import YOLO_Detection, drawPolygons, label_detection

# --- File paths ---
ROI_PICKLE = "Space_ROIs.pkl"
VIDEO_PATH = "input_video/parking_space.mp4"
MODEL_PATH = "yolo11n.pt"  # ✅ you can use yolov8n.pt or yolov11n.pt (after pip upgrade)


# --- Helper to scale polygons if video resolution differs from ROI reference ---
def scale_polygons(polys, ref_size, cur_size):
    ref_w, ref_h = ref_size
    cur_h, cur_w = cur_size
    sx, sy = cur_w / ref_w, cur_h / ref_h
    return [[(int(x * sx), int(y * sy)) for (x, y) in poly] for poly in polys]


# --- Model setup (CPU) ---
device = torch.device('cpu')
model = YOLO(MODEL_PATH)
model.to(device)

# --- Load polygons and reference size (compatible with new and old pickle formats) ---
with open(ROI_PICKLE, 'rb') as f:
    data = pickle.load(f)
    if isinstance(data, dict):
        posList_raw = data["polygons"]
        ref_size = data["size"]
    else:
        posList_raw = data
        ref_size = None

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
cur_frame = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur_frame += 1

        cur_h, cur_w = frame.shape[:2]
        if ref_size is None:
            ref_size = (cur_w, cur_h)

        posList = scale_polygons(posList_raw, ref_size, (cur_h, cur_w))

        # --- YOLO detection ---
        boxes, classes, names = YOLO_Detection(model, frame, conf=0.35)

        # --- Get car centers for polygon occupancy ---
        detection_points = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for (x1, y1, x2, y2) in boxes]

        # --- Draw polygons and get status ---
        frame, occupied_count, slot_status = drawPolygons(frame, posList, detection_points=detection_points)
        available_count = len(posList) - occupied_count
        occupancy_percent = (occupied_count / len(posList)) * 100 if len(posList) > 0 else 0

        # --- Top-right Parking Status Panel ---
        panel_x, panel_y = frame.shape[1] - 300, 10
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 290, panel_y + 150), (40, 40, 40), -1)
        cv2.putText(frame, "PARKING STATUS", (panel_x + 10, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total: {len(posList)}", (panel_x + 10, panel_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Occupied: {occupied_count}", (panel_x + 10, panel_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Available: {available_count}", (panel_x + 10, panel_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Occupancy: {occupancy_percent:.1f}%", (panel_x + 10, panel_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # --- Progress bar (frame number indicator) ---
        bar_x0, bar_y0, bar_w, bar_h = panel_x + 150, panel_y + 125, 120, 15
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (100, 100, 100), 1)
        cv2.rectangle(frame, (bar_x0, bar_y0),
                      (bar_x0 + int(bar_w * occupancy_percent / 100), bar_y0 + bar_h),
                      (0, 0, 255) if occupancy_percent > 50 else (0, 255, 0), -1)
        cv2.putText(frame, f"Frame: {cur_frame}/{frame_count}",
                    (panel_x + 80, panel_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # --- Show detections (optional bounding boxes) ---
        for (x1, y1, x2, y2), cls in zip(boxes, classes):
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            in_poly = any(cv2.pointPolygonTest(np.array(p, np.int32), center, False) >= 0 for p in posList)
            name = names[int(cls)]
            color = (0, 0, 255) if in_poly else (0, 255, 0)
            label_detection(frame, text=str(name), tbox_color=color, left=x1, top=y1, bottom=x2, right=y2)

        cv2.imshow("AI Parking Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()