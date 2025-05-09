import cv2
from PIL import Image

cap = cv2.VideoCapture(0)
frames = []

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated

target_size = (640, 360)
ret, frame = cap.read()
cv2.imshow("Resized Frame", frame)
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
tracker = cv2.legacy.TrackerMedianFlow_create()
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    ret, bbox = tracker.update(frame)
    if ret:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    frame = cv2.resize(frame, target_size)
    frame = rotate_image(frame, 0)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frames.append(pil_image)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total frames to save: {len(frames)}")
if len(frames) > 1:
    frames[0].save('Traking_output.gif', save_all=True, append_images=frames[1:], duration=50, loop=0)
else:
    print("Not enough frames to save GIF.")
