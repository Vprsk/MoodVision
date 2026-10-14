from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

from contracts.schemas import BoundingBox, FaceDetection


class YoloFaceDetector:
    """Детектор и трекер лиц на основе YOLO."""

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.3,
        verbose: bool = False,
        device: str | int = 0,
        tracker: str | Path = "trackers/tracktrack.yaml",
    ) -> None:
        self.model: YOLO = YOLO(model_path)
        self.conf: float = conf
        self.verbose: bool = verbose
        self.device: str | int = device
        self.tracker: str | Path = tracker

    def _preprocess(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Подготавливает кадр к инференсу."""
        if frame is None or frame.size == 0:
            raise ValueError("Получен пустой кадр")

        return frame

    def _infer(self, frame: NDArray[np.uint8]) -> Results | None:
        """Запускает трекинг для одного кадра и возвращает первый Result."""
        results: Any = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            verbose=self.verbose,
            device=self.device,
            stream=True,
        )

        return next(results, None)

    def _postprocess(self, result: Results | None) -> list[FaceDetection]:
        """Преобразует Result YOLO в список FaceDetection."""
        if result is None or result.boxes is None:
            return []

        tracked_faces: list[FaceDetection] = []
        boxes = result.boxes

        track_ids: list[int | None] = (
            boxes.id.tolist()
            if boxes.id is not None
            else [None] * len(boxes)
        )

        for xyxy, confidence, track_id in zip(
            boxes.xyxy,
            boxes.conf,
            track_ids,
        ):
            x1, y1, x2, y2 = map(int, xyxy.tolist())

            tracked_faces.append(
                FaceDetection(
                    track_id=int(track_id) if track_id is not None else None,
                    bbox=BoundingBox(
                        x_min=x1,
                        y_min=y1,
                        x_max=x2,
                        y_max=y2,
                    ),
                    emotion=None,
                    face_confidence=float(confidence),
                    emotion_confidence=None,
                )
            )

        return tracked_faces

    def detect(self, frame: NDArray[np.uint8]) -> list[FaceDetection]:
        """Выполняет полный пайплайн детекции лиц."""
        prepared_frame = self._preprocess(frame)
        result = self._infer(prepared_frame)

        return self._postprocess(result)