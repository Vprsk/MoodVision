"""Схемы данных контракта между моделями распознавания и приложением.

Любая модель обязана возвращать строго валидный FrameResult (см. contracts.base).
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import numpy as np




class Emotion(str, Enum):
    """Строгий перечень эмоций.

    Значения совпадают с метками модели trpakov/vit-face-expression.
    UNKNOWN — служебный фоллбек, когда уверенность классификатора
    ниже порога (порог задаётся в адаптере модели, не в схеме).
    """

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISE = "surprise"  # именно "surprise" — так отдаёт ViT-модель
    FEAR = "fear"
    DISGUST = "disgust"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """BBox в координатах углов исходного кадра: frame[y_min:y_max, x_min:x_max]."""

    x_min: int = Field(ge=0, description="Левая граница лица")
    y_min: int = Field(ge=0, description="Верхняя граница лица")
    x_max: int = Field(gt=0, description="Правая граница лица")
    y_max: int = Field(gt=0, description="Нижняя граница лица")


class FaceDetection(BaseModel):
    """Результат распознавания одного лица в кадре."""

    track_id: Optional[int] = Field(
        default=None,
        description="ID трека от ByteTrack; None — если модель работает без трекинга",
    )
    bbox: BoundingBox = Field(description="Координаты лица")
    emotion: Optional[Emotion] = Field(description="Определённая эмоция")
    face_confidence: float = Field(
        ge=0.0, le=1.0, description="Уверенность детекции лица (YOLO)"
    )
    emotion_confidence: Optional[float] = Field(
        ge=0.0, le=1.0, description="Уверенность классификатора эмоций"
    )


class FrameResult(BaseModel):
    """Результат обработки одного кадра — единица обмена между моделью и приложением."""

    frame_id: int = Field(ge=0, description="Порядковый номер кадра")
    detections: List[FaceDetection] = Field(default_factory=list)


"""Контракт для всех моделей распознавания лиц и эмоций."""

class BaseFaceEmotionModel(ABC):
    """Базовый класс (контракт) для всех моделей распознавания.

    Любая новая модель (YOLO+ViT, MediaPipe, другая нейросеть) обязана
    реализовать predict() и возвращать строго валидный FrameResult.
    """

    @abstractmethod
    def predict(self, frame: np.ndarray) -> FrameResult:
        """Принимает номер кадра и сам кадр, возвращает результат в формате FrameResult."""
    contract_version: int = Field(
        default=1, description="Версия контракта (для будущего V2: возраст/очки и т.п.)"
    )
    frame_id: int = Field(ge=0, description="Порядковый номер кадра")
    detections: List[FaceDetection] = Field(default_factory=list)
