"""Контракт для всех моделей распознавания лиц и эмоций."""

from abc import ABC, abstractmethod

import numpy as np

from contracts.schemas import FrameResult


class BaseFaceEmotionModel(ABC):
    """Базовый класс (контракт) для всех моделей распознавания.

    Любая новая модель (YOLO+ViT, MediaPipe, другая нейросеть) обязана
    реализовать predict() и возвращать строго валидный FrameResult.
    """

    @abstractmethod
    def __init__(self, model_path: str) -> None:
        pass

    @abstractmethod
    def predict(self, frame_id: int, frame: np.ndarray) -> FrameResult:
        """Принимает номер кадра и сам кадр, возвращает результат в формате FrameResult."""