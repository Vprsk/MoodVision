# Video Analysis: распознавание лиц и эмоций

Проект обрабатывает видеопоток из файла, веб-камеры или RTSP-источника: находит лица с помощью YOLO, назначает им трек-ID и определяет эмоцию для каждого лица.

## Структура проекта

```text
video-analys/
├── cam_reader.py          # Захват кадров из видео/камеры
├── face_detect.py         # Детекция и трекинг лиц YOLO
├── mood_class.py          # Классификация эмоций
├── mood_vision.py         # Общий пайплайн: лица + эмоции
├── contracts/
│   └── schemas.py         # Pydantic-схемы и базовый контракт модели
├── checkpoints/
│   └── yolov8n-face.pt    # YOLO-модель детекции лиц
├── trackers/
│   └── tracktrack.yaml    # Вариант трекера, рекомендуемый YOLO
│   └── bytetrack.yaml     # Вариант трекера, рекомендуемый YOLO
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

Поместите свою YOLO-модель детекции лиц в:

```text
checkpoints/
```

Или используйте уже готовую
---
Для эмоций по умолчанию используется Hugging Face-модель:

```text
trpakov/vit-face-expression
```

При первом запуске она будет скачана из Hugging Face Hub. 
---
Поместите свою конфигурацию трекера детекции лиц в:

```text
trackers/
```
Или используйте уже готовую

## Основной API
Большинство взаимодействий происходит через объект класса MoodVision
Создайте объект класса MoodVision и обратитесь к его методу predict(), для получения результата

#### Параметры `MoodVision`

| Параметр | По умолчанию | Описание |
|---|---:|---|
| `face_model_path` | `checkpoints/yolov8n-face.pt` | Путь к YOLO-модели детекции лиц |
| `emotion_model_path` | `trpakov/vit-face-expression` | Hugging Face-модель классификации эмоций |
| `face_conf` | `0.3` | Минимальная уверенность детекции лица |
| `emotion_conf` | `0.5` | Порог уверенности эмоции |
| `yolo_device` | `0` | Где производится инференс yolo модели `0` — первый GPU, `"cpu"` — процессор |
| `hf_device` | `0` | Где производится инференс hugging_face модели `0` — первый GPU, `"cpu"` — процессор |
| `tracker` | `bytetrack.yaml` | Конфигурация трекера YOLO |
| `verbose` | `False` | Логирование YOLO-модели |

## Формат результата

`MoodVision.predict()` возвращает объект `FrameResult`:

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


