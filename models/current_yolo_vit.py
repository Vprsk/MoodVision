"""Адаптер текущего пайплайна: YOLOv8-face (детекция + ByteTrack) + ViT (эмоции).

Обёртка логики прототипа try.py в контракт BaseFaceEmotionModel.
Вся «грязь» конкретных моделей (свои bbox, метки, confidence) изолирована здесь.
"""

import cv2
import numpy as np
import torch
from transformers import pipeline
from ultralytics import YOLO

from contracts.base import BaseFaceEmotionModel
from contracts.schemas import BoundingBox, Emotion, FaceDetection, FrameResult


class CurrentYoloVitModel(BaseFaceEmotionModel):
    """Текущий продакшн-пайплайн, приведённый к контракту."""

    def __init__(
        self,
        model_path: str = "model/yolov8n-face.pt",
        tracker_config: str = "configs/bytetrack.yaml",
        conf_threshold: float = 0.45,
        emotion_conf_threshold: float = 0.35,
        yolo_imgsz: int = 416,
        analyze_every_n_frames: int = 5,
        frame_scale: float = 0.5,
    ) -> None:
        # --- Выбор устройства (как в прототипе) ---
        if torch.cuda.is_available():
            self.device = 0
            self.yolo_device = 0
            self.torch_dtype = torch.float16
            print(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
        else:
            self.device = -1
            self.yolo_device = "cpu"
            self.torch_dtype = torch.float32
            print("CUDA недоступна, используется CPU")

        # --- Параметры ---
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.emotion_conf_threshold = emotion_conf_threshold
        self.yolo_imgsz = yolo_imgsz
        self.analyze_every_n_frames = analyze_every_n_frames
        self.frame_scale = frame_scale

        # --- Модели ---
        self.face_model = YOLO(model_path)
        self.emotion_classifier = pipeline(
            "image-classification",
            model="trpakov/vit-face-expression",
            device=self.device,
            dtype=self.torch_dtype,
        )

        # --- Кэш эмоций по track_id: пересчитываем каждые N кадров ---
        self._last_results: dict[int, tuple[str, float]] = {}

    def _classify_emotion(self, face_bgr: np.ndarray) -> tuple[str, float]:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        predictions = self.emotion_classifier(face_rgb)
        best = predictions[0]
        emotion = best["label"]
        confidence = float(best["score"])
        if confidence < self.emotion_conf_threshold:
            emotion = "unknown"
        return emotion, confidence

    def predict(self, frame_id: int, frame: np.ndarray) -> FrameResult:
        # Уменьшаем кадр перед YOLO
        small = cv2.resize(
            frame,
            None,
            fx=self.frame_scale,
            fy=self.frame_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        # Детекция и трекинг
        results = self.face_model.track(
            small,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            imgsz=self.yolo_imgsz,
            device=self.yolo_device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        detections: list[FaceDetection] = []

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().tolist()
            confidences = boxes.conf.tolist()
            bboxes = boxes.xyxy.int().tolist()

            h, w = frame.shape[:2]

            for track_id, face_conf, raw_bbox in zip(track_ids, confidences, bboxes):
                # Возвращаем координаты к исходному разрешению
                x1, y1, x2, y2 = raw_bbox
                x1 = int(x1 / self.frame_scale)
                y1 = int(y1 / self.frame_scale)
                x2 = int(x2 / self.frame_scale)
                y2 = int(y2 / self.frame_scale)

                # Обрезаем по границам кадра
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Эмоцию считаем не каждый кадр (кэш по track_id)
                if frame_id % self.analyze_every_n_frames == 0:
                    face_crop = frame[y1:y2, x1:x2]
                    emotion, emotion_conf = self._classify_emotion(face_crop)
                    self._last_results[track_id] = (emotion, emotion_conf)
                else:
                    emotion, emotion_conf = self._last_results.get(
                        track_id, ("unknown", 0.0)
                    )

                detections.append(
                    FaceDetection(
                        track_id=track_id,
                        bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                        emotion=Emotion(emotion),
                        face_confidence=float(face_conf),
                        emotion_confidence=float(emotion_conf),
                    )
                )

        return FrameResult(frame_id=frame_id, detections=detections)
