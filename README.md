# Video Analysis: распознавание лиц и эмоций

Проект обрабатывает видеопоток из файла, веб-камеры или RTSP-источника: находит лица с помощью YOLO, назначает им трек-ID и определяет эмоцию для каждого лица. Результат отображается в окне OpenCV. [docs.ultralytics](https://docs.ultralytics.com/quickstart)

## Возможности

- Захват кадров из локального видео, веб-камеры или RTSP.
- Детекция и трекинг лиц через YOLO.
- Классификация эмоций через Hugging Face `image-classification`.
- Отрисовка bounding box, `track_id`, эмоции и confidence.
- Ленивая обработка кадров через генератор.

## Структура проекта

```text
video-analys/
├── cam_reader.py          # Захват кадров из видео/камеры
├── face_detect.py         # Детекция и трекинг лиц YOLO
├── mood_class.py          # Классификация эмоций
├── mood_vision.py         # Общий пайплайн: лица + эмоции
├── pipeline.py            # Связка источника кадров и модели, отображение
├── contracts/
│   └── schemas.py         # Pydantic-схемы и базовый контракт модели
├── checkpoints/
│   └── yolov8n-face.pt    # YOLO-модель детекции лиц
└── requirements.txt
```

## Установка

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

## Модели

Поместите YOLO-модель детекции лиц в:

```text
checkpoints/yolov8n-face.pt
```

Модель должна быть обучена на классе `face`. Проверить классы можно так:

```python
from ultralytics import YOLO

model = YOLO("checkpoints/yolov8n-face.pt")
print(model.names)
```

Для эмоций по умолчанию используется Hugging Face-модель:

```text
trpakov/vit-face-expression
```

При первом запуске она будет скачана из Hugging Face Hub. 

## Быстрый запуск

```bash
python pipeline.py
```

По умолчанию обрабатывается файл:

```python
CameraReader(source="test.mp4")
```

Чтобы использовать веб-камеру, замените:

```python
camera = CameraReader(source=0)
```

Для RTSP:

```python
camera = CameraReader(source="rtsp://user:password@host:554/stream")
```

## Пример использования

```python
from cam_reader import CameraReader
from mood_vision import MoodVision
from pipeline import Pipeline


camera = CameraReader(source="test.mp4")

model = MoodVision(
    face_model_path="checkpoints/yolov8n-face.pt",
    emotion_model_path="trpakov/vit-face-expression",
    face_conf=0.3,
    emotion_conf=0.5,
    device=0,
    tracker="bytetrack.yaml",
)

pipeline = Pipeline(camera, model)
pipeline.run()
```

Во время работы нажмите `q` или `Esc`, чтобы остановить обработку.

## Параметры `MoodVision`

| Параметр | По умолчанию | Описание |
|---|---:|---|
| `face_model_path` | `checkpoints/yolov8n-face.pt` | Путь к YOLO-модели детекции лиц |
| `emotion_model_path` | `trpakov/vit-face-expression` | Hugging Face-модель классификации эмоций |
| `face_conf` | `0.3` | Минимальная уверенность детекции лица |
| `emotion_conf` | `0.5` | Порог уверенности эмоции |
| `device` | `0` | `0` — первый GPU, `"cpu"` — процессор |
| `tracker` | `bytetrack.yaml` | Конфигурация трекера YOLO |

## Формат результата

`MoodVision.predict()` возвращает `FrameResult`:

```python
FrameResult(
    frame_id=0,
    detections=[
        FaceDetection(
            track_id=1,
            bbox=BoundingBox(
                x_min=639,
                y_min=81,
                x_max=1367,
                y_max=862,
            ),
            emotion="happy",
            face_confidence=0.83,
            emotion_confidence=0.91,
        )
    ]
)
```

| Поле | Описание |
|---|---|
| `frame_id` | Номер кадра в потоке |
| `detections` | Список найденных лиц |
| `track_id` | ID лица от трекера или `None` |
| `bbox` | Координаты рамки лица |
| `emotion` | Распознанная эмоция |
| `face_confidence` | Уверенность детектора лиц |
| `emotion_confidence` | Уверенность классификатора эмоций |

## Обработка только кадров без отображения

```python
from cam_reader import CameraReader
from mood_vision import MoodVision

camera = CameraReader("test.mp4")

model = MoodVision(
    face_model_path="checkpoints/yolov8n-face.pt",
    device=0,
)

for frame_id, frame in camera.get_frames():
    result = model.predict(frame_id=frame_id, frame=frame)

    print(f"Frame {frame_id}: {len(result.detections)} faces")

    for face in result.detections:
        print(
            face.track_id,
            face.bbox,
            face.emotion,
            face.face_confidence,
        )
```

