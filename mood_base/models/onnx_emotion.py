"""Классификатор эмоций на базе ONNX-модели (poster_v2.onnx).

Вход:  [1, 3, 224, 224] RGB-изображение лица.
Выход: [1, 7] логиты по 7 эмоциям.

Предобработка (строго по требованиям POSTER V2):
  • Изображение уменьшается до 224x224 (RGB).
  • Пиксели нормируются: (pixel / 255.0 - mean) / std
    со средними ImageNet: mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225].

⚠️ ВАЖНО: текущий файл poster_v2.onnx не дифференцирует эмоции
(выдаёт одинаковые логиты для любого лица). Причина — модель была
экспортирована без model.eval(), поэтому BatchNorm running-статистика
была перезаписана случайными значениями из dummy-входа, а также возможна
проблема с загрузкой state_dict (префикс module. при DataParallel).
Модель требует переобучения / переэкспорта с корректным eval() и
очисткой ключей весов перед использованием в продакшене.

Реализует контракт BaseFaceEmotionModel.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from contracts.base import BaseFaceEmotionModel
from contracts.schemas import (
    BoundingBox,
    Emotion,
    FaceDetection,
    FrameResult,
)

try:
    from app.core.logging import get_logger

    logger = get_logger("onnx_emotion")
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("onnx_emotion")

EMOTION_LABELS: list[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

EMOTION_CONF_THRESHOLD = 0.35


class OnnxEmotionClassifier(BaseFaceEmotionModel):
    """YOLOv8-face (детекция) + ONNX-классификатор эмоций (poster_v2.onnx)."""

    def __init__(
        self,
        model_path: str = "checkpoints/poster_v2.onnx",
        face_model_path: str = "mood_base/model/yolov8n-face.pt",
        conf_threshold: float = 0.3,
        emotion_conf_threshold: float = EMOTION_CONF_THRESHOLD,
        yolo_device: str | int = 0,
        tracker: str = "trackers/tracktrack.yaml",
        yolo_imgsz: int = 416,
        frame_scale: float = 0.5,
        analyze_every_n_frames: int = 1,
        verbose: bool = False,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.emotion_conf_threshold = emotion_conf_threshold
        self.yolo_device = yolo_device
        self.tracker = tracker
        self.yolo_imgsz = yolo_imgsz
        self.frame_scale = frame_scale
        self.analyze_every_n_frames = analyze_every_n_frames
        self.verbose = verbose

        self.face_model = YOLO(face_model_path)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        logger.info(
            "ONNX эмоции загружены: %s (input=%s, output=%s)",
            model_path,
            self.input_name,
            self.output_name,
        )

        self._last_results: dict[int, tuple[Emotion, float]] = {}
        self._frame_counter = 0

    def _preprocess_face(
        self, face_bgr: NDArray[np.uint8], target_size: int = 224
    ) -> NDArray[np.float32]:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(
            face_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )
        face_chw = face_resized.transpose(2, 0, 1).astype(np.float32)
        return face_chw[np.newaxis, ...]

    def _classify_emotion(
        self, face_bgr: NDArray[np.uint8]
    ) -> tuple[Emotion, float]:
        input_tensor = self._preprocess_face(face_bgr)
        outputs = self.ort_session.run(
            [self.output_name], {self.input_name: input_tensor}
        )
        logits = outputs[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx])
        best_label = EMOTION_LABELS[best_idx]

        if best_conf < self.emotion_conf_threshold:
            return Emotion.UNKNOWN, best_conf

        return Emotion(best_label), best_conf

    def _crop_face(
        self, frame: NDArray[np.uint8], bbox: BoundingBox
    ) -> NDArray[np.uint8]:
        height, width = frame.shape[:2]
        x1 = max(0, bbox.x_min)
        y1 = max(0, bbox.y_min)
        x2 = min(width, bbox.x_max)
        y2 = min(height, bbox.y_max)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Некорректный bounding box лица")
        return frame[y1:y2, x1:y2].copy()

    def _detect_faces(self, frame: NDArray[np.uint8]):
        small = cv2.resize(
            frame,
            None,
            fx=self.frame_scale,
            fy=self.frame_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        results = self.face_model.track(
            small,
            persist=True,
            tracker=self.tracker,
            conf=self.conf_threshold,
            imgsz=self.yolo_imgsz,
            device=self.yolo_device,
            verbose=self.verbose,
        )
        return results[0]

    def _process_one_frame(
        self,
        frame_id: int,
        frame: NDArray[np.uint8],
        result: Any,
    ) -> FrameResult:
        detections: list[FaceDetection] = []

        if result is None or result.boxes is None:
            return FrameResult(frame_id=frame_id, detections=[])

        boxes = result.boxes
        track_ids: list[int | None] = (
            boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        )

        current_ids: set[int] = set()

        for xyxy, confidence, track_id in zip(
            boxes.xyxy, boxes.conf, track_ids
        ):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            scale = 1.0 / self.frame_scale
            x1, y1, x2, y2 = (
                int(x1 * scale),
                int(y1 * scale),
                int(x2 * scale),
                int(y2 * scale),
            )

            tid = int(track_id) if track_id is not None else None
            if tid is not None:
                current_ids.add(tid)

            bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

            emotion = Emotion.UNKNOWN
            emotion_conf = 0.0
            use_cache = (
                tid is not None
                and self.analyze_every_n_frames > 1
                and tid in self._last_results
                and self._frame_counter % self.analyze_every_n_frames != 0
            )
            if use_cache:
                emotion, emotion_conf = self._last_results[tid]
            else:
                try:
                    face_crop = self._crop_face(frame, bbox)
                    emotion, emotion_conf = self._classify_emotion(face_crop)
                    if tid is not None:
                        self._last_results[tid] = (emotion, emotion_conf)
                except ValueError:
                    emotion = Emotion.UNKNOWN
                    emotion_conf = 0.0

            detections.append(
                FaceDetection(
                    track_id=tid,
                    bbox=bbox,
                    emotion=emotion,
                    face_confidence=float(confidence),
                    emotion_confidence=emotion_conf,
                )
            )

        for stale_tid in [
            tid for tid in self._last_results if tid not in current_ids
        ]:
            del self._last_results[stale_tid]

        self._frame_counter += 1
        return FrameResult(frame_id=frame_id, detections=detections)

    def predict(
        self, frame_id: int, frame: NDArray[np.uint8]
    ) -> FrameResult:
        result = self._detect_faces(frame)
        return self._process_one_frame(frame_id, frame, result)

    def predict_batch(
        self, frames: list[tuple[int, NDArray[np.uint8]]]
    ) -> list[FrameResult]:
        if not frames:
            return []

        small_frames = [
            cv2.resize(
                f,
                None,
                fx=self.frame_scale,
                fy=self.frame_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            for _, f in frames
        ]

        try:
            results = self.face_model.track(
                small_frames,
                persist=True,
                tracker=self.tracker,
                conf=self.conf_threshold,
                imgsz=self.yolo_imgsz,
                device=self.yolo_device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Батч YOLO не удался (%s), перехожу на по-кадровый режим", exc
            )
            return [self.predict(frame_id, frame) for frame_id, frame in frames]

        if not isinstance(results, list):
            results = [results]

        return [
            self._process_one_frame(frame_id, frame, res)
            for (frame_id, frame), res in zip(frames, results)
        ]