from typing import Any

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from transformers import pipeline

from contracts.schemas import Emotion, FaceDetection


class HuggingFaceEmotionClassifier:
    """Классифицирует эмоцию для одного найденного лица."""

    def __init__(
        self,
        model_path: str = "trpakov/vit-face-expression",
        device: str = 0,
    ) -> None:
        self.pipeline = pipeline(
            "image-classification",
            model=model_path,
            device=device,
        )

    def _crop_face(
        self,
        frame: NDArray[np.uint8],
        bbox,
    ) -> NDArray[np.uint8]:
        """Вырезает лицо по bounding box."""
        height, width = frame.shape[:2]

        x1 = max(0, bbox.x_min)
        y1 = max(0, bbox.y_min)
        x2 = min(width, bbox.x_max)
        y2 = min(height, bbox.y_max)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Некорректный bounding box лица")

        return frame[y1:y2, x1:x2]

    def _infer(
        self,
        face_image: NDArray[np.uint8],
    ) -> list[dict[str, Any]]:
        """Запускает классификацию эмоции."""
        rgb_image = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)

        return self.pipeline(rgb_image)

    def _postprocess(
        self,
        face: FaceDetection,
        predictions: list[dict[str, Any]],
    ) -> FaceDetection:
        """Выбирает наиболее вероятную эмоцию."""
        if not predictions:
            return face

        best = predictions[0]

        return face.model_copy(
            update={
                "emotion": Emotion(best["label"].lower()),
                "emotion_confidence": float(best["score"]),
            }
        )

    def predict(
        self,
        frame: NDArray[np.uint8],
        face: FaceDetection,
    ) -> FaceDetection:
        """Определяет эмоцию для одного лица."""
        face_image = self._crop_face(frame, face.bbox)
        predictions = self._infer(face_image)

        return self._postprocess(face, predictions)