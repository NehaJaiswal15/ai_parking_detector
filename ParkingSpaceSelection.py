import cv2
import pickle
import numpy as np

ROI_PICKLE = "Space_ROIs.pkl"
REF_IMAGE = "ROI_Reference.png"

# --- Load existing polygons (supports both old and new formats) ---
try:
    with open(ROI_PICKLE, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            posList = data.get("polygons", [])
        else:
            posList = data
    print(f"✅ Loaded {len(posList)} saved parking slots.")
except FileNotFoundError:
    posList = []
    print("⚠️ No existing slots found. Starting fresh...")

polygon_points = []

def mouseClick(event, x, y, flags, params):
    global polygon_points, posList

    # Left-click: mark polygon corners
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"📍 Point added: {x}, {y}")
        if len(polygon_points) == 4:
            posList.append(polygon_points.copy())
            save()
            print(f"✅ Added slot #{len(posList)}")
            polygon_points.clear()

    # Right-click: remove a slot
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, polygon in enumerate(posList):
            if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (x, y), False) >= 0:
                posList.pop(i)
                save()
                print(f"❌ Removed slot #{i+1}")
                break

def save():
    """Save polygons + image size to pickle."""
    img = cv2.imread(REF_IMAGE)
    h, w = img.shape[:2]
    with open(ROI_PICKLE, 'wb') as f:
        pickle.dump({"size": (w, h), "polygons": posList}, f)
    print(f"💾 Saved {len(posList)} slots to {ROI_PICKLE}")

# --- Display setup ---
img0 = cv2.imread(REF_IMAGE)
if img0 is None:
    raise FileNotFoundError(f"❌ Could not load {REF_IMAGE}. Make sure it exists in the folder.")

h0, w0 = img0.shape[:2]
cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("Image", w0, h0)  # ensure 1:1 pixels

print("🖱 Left-click 4 points per slot | Right-click inside a slot to delete | Press 'Q' to save & quit")

while True:
    img = img0.copy()

    # Draw saved polygons
    for idx, polygon in enumerate(posList, start=1):
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        # Label each slot number
        cx = int(np.mean([p[0] for p in polygon]))
        cy = int(np.mean([p[1] for p in polygon]))
        cv2.putText(img, str(idx), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show in-progress points
    for point in polygon_points:
        cv2.circle(img, point, 5, (0, 255, 0), -1)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)

    # Quit and save on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        save()
        break

cv2.destroyAllWindows()
print("✅ Polygon marking session ended. All data saved.")