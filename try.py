import time

import cv2
import torch
from ultralytics import YOLO
from transformers import pipeline

# ==============================
# Настройки
# ==============================

FACE_MODEL_PATH = "chekpoints/yolov8n-face.pt"
VIDEO_SOURCE = "test.mp4"  # Вставьте сюда ссылку на ресурс или путь к видео

CONF_THRESHOLD = 0.45
EMOTION_CONF_THRESHOLD = 0.35

YOLO_IMGSZ = 416
ANALYZE_EVERY_N_FRAMES = 5
FRAME_SCALE = 0.5

# ==============================
# Выбор устройства
# ==============================

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    DEVICE = 0
    YOLO_DEVICE = 0
    TORCH_DTYPE = torch.float16
    print(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = -1
    YOLO_DEVICE = "cpu"
    TORCH_DTYPE = torch.float32
    print("CUDA недоступна, используется CPU")

# ==============================
# Загрузка моделей
# ==============================

face_model = YOLO(FACE_MODEL_PATH)

emotion_classifier = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression",
    device=DEVICE,
    dtype=TORCH_DTYPE,
)

# ==============================
# Кэш последних эмоций
# ==============================

last_results = {}

# ==============================
# Классификация эмоции
# ==============================

def classify_emotion(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    predictions = emotion_classifier(face_rgb)
    best = predictions[0]

    emotion = best["label"]
    confidence = float(best["score"])

    if confidence < EMOTION_CONF_THRESHOLD:
        emotion = "unknown"

    return emotion, confidence

# ==============================
# Отрисовка
# ==============================

def draw_result(frame, bbox, track_id, emotion, confidence):
    x1, y1, x2, y2 = bbox

    color = (0, 255, 0)

    if emotion == "unknown":
        color = (0, 165, 255)
    elif emotion == "happy":
        color = (0, 255, 255)
    elif emotion == "sad":
        color = (255, 0, 0)
    elif emotion == "angry":
        color = (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID {track_id} | {emotion} {confidence:.2f}"

    cv2.putText(
        frame,
        label,
        (x1, max(15, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

# ==============================
# Основной цикл
# ==============================

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник: {VIDEO_SOURCE}")

    frame_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Видеопоток завершён или кадр не получен")
            break

        frame_id += 1
        loop_start = time.perf_counter()

        # Уменьшаем кадр перед YOLO
        small = cv2.resize(
            frame,
            None,
            fx=FRAME_SCALE,
            fy=FRAME_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )

        # Детекция и трекинг на GPU
        results = face_model.track(
            small,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD,
            imgsz=YOLO_IMGSZ,
            device=YOLO_DEVICE,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().tolist()
            confidences = boxes.conf.tolist()
            bboxes = boxes.xyxy.int().tolist()

            for track_id, face_conf, raw_bbox in zip(
                track_ids, confidences, bboxes
            ):
                # Возвращаем координаты к исходному разрешению
                x1, y1, x2, y2 = raw_bbox

                x1 = int(x1 / FRAME_SCALE)
                y1 = int(y1 / FRAME_SCALE)
                x2 = int(x2 / FRAME_SCALE)
                y2 = int(y2 / FRAME_SCALE)

                h, w = frame.shape[:2]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Эмоцию считаем не каждый кадр
                if frame_id % ANALYZE_EVERY_N_FRAMES == 0:
                    face_crop = frame[y1:y2, x1:x2]

                    emotion_start = time.perf_counter()
                    emotion, emotion_conf = classify_emotion(face_crop)
                    emotion_time = time.perf_counter() - emotion_start

                    last_results[track_id] = (emotion, emotion_conf)
                else:
                    emotion, emotion_conf = last_results.get(
                        track_id,
                        ("unknown", 0.0),
                    )

                draw_result(
                    frame,
                    (x1, y1, x2, y2),
                    track_id,
                    emotion,
                    emotion_conf,
                )

                print(
                    {
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "face_conf": round(face_conf, 3),
                        "bbox": [x1, y1, x2, y2],
                        "emotion": emotion,
                        "emotion_conf": round(emotion_conf, 3),
                    }
                )

        fps = 1.0 / max(time.perf_counter() - loop_start, 1e-6)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Face emotion recognition (CUDA)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()