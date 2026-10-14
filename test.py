import cv2 as cv
from ultralytics import YOLO

model = YOLO("checkpoints/yolov8n-face.pt")

print("classes:", model.names)

cap = cv.VideoCapture("test.mp4")

ok, frame = cap.read()
print("frame read:", ok)
print("frame shape:", None if frame is None else frame.shape)

if ok:
    results = model.predict(frame, conf=0.1, verbose=True)
    boxes = results[0].boxes

    print("detections:", len(boxes))
    print("boxes:", boxes.xyxy)
    print("conf:", boxes.conf)

    cv.imshow("debug", results[0].plot())
    cv.waitKey(0)

cap.release()
cv.destroyAllWindows()