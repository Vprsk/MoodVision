"""Схемы данных контракта между моделями распознавания и приложением.

Любая модель обязана возвращать строго валидный FrameResult (см. contracts.base).
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


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


class BodyPostureTag(str, Enum):
    """Биомеханические маркеры позы клиента (Уровни 1 и 2 анализа языка тела)."""

    NEUTRAL_POSTURE = "neutral_posture"      # Спокойная открытая поза
    AGITATED_GESTURES = "agitated_gestures"  # Активное разведение/взмахи руками при споре
    HANDS_ON_HIPS = "hands_on_hips"          # Руки в боки (претензия, агрессивное ожидание)
    ARMS_CROSSED = "arms_crossed"            # Скрещенные руки на груди (закрытое недовольство)
    HANDS_TO_HEAD = "hands_to_head"          # Руки у головы/лица (стресс, возмущение, фрустрация)
    SLOUCHED_HEAD_DOWN = "slouched_head_down"  # Опущенная голова и сутулость (подавленность, грусть)
    LEANING_FORWARD = "leaning_forward"      # Резкий наклон корпуса вперёд через стойку


class BoundingBox(BaseModel):
    """BBox в координатах углов исходного кадра: frame[y_min:y_max, x_min:x_max]."""

    x_min: int = Field(ge=0, description="Левая граница")
    y_min: int = Field(ge=0, description="Верхняя граница")
    x_max: int = Field(gt=0, description="Правая граница")
    y_max: int = Field(gt=0, description="Нижняя граница")

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x_min + self.x_max)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y_min + self.y_max)


class Keypoint2D(BaseModel):
    """Одна скелетная точка COCO (17 точек от YOLO-pose)."""

    x: float
    y: float
    conf: float = Field(ge=0.0, le=1.0)


class FaceDetection(BaseModel):
    """Результат распознавания одного лица (или прокси-рамки человека) в кадре."""

    track_id: Optional[int] = Field(
        default=None,
        description="ID трека от ByteTrack; None — если модель работает без трекинга",
    )
    bbox: BoundingBox = Field(description="Координаты рамки")
    emotion: Emotion = Field(
        default=Emotion.UNKNOWN, description="Определённая эмоция"
    )
    face_confidence: float = Field(
        ge=0.0, le=1.0, description="Уверенность детекции (YOLO)"
    )
    emotion_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Уверенность классификатора эмоций"
    )
    passed_quality_gate: bool = Field(
        default=True, description="Прошло ли лицо фильтр разрешения и чёткости"
    )


class BodyDetection(BaseModel):
    """Результат 3-уровневого анализа языка тела человека (Сцены 2 и 3)."""

    track_id: Optional[int] = Field(default=None, description="Стабильный ID человека")
    bbox: BoundingBox = Field(description="Координаты рамки тела")
    body_confidence: float = Field(ge=0.0, le=1.0, description="Уверенность YOLO-pose")
    keypoints: List[Keypoint2D] = Field(
        default_factory=list, description="17 ключевых точек скелета COCO"
    )
    kinetic_energy: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Уровень 1: нормированная скорость рук"
    )
    posture_tag: BodyPostureTag = Field(
        default=BodyPostureTag.NEUTRAL_POSTURE,
        description="Уровень 2: биомеханический тег позы",
    )
    body_emotion: Emotion = Field(
        default=Emotion.UNKNOWN,
        description="Уровень 3: эмоция по языку тела (нейросеть + вектор позы)",
    )
    body_emotion_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Уверенность классификатора языка тела"
    )
    face: Optional[FaceDetection] = Field(
        default=None, description="Привязанное лицо (используется в Сцене 2)"
    )


class AudioSegmentResult(BaseModel):
    """Результат анализа аудиоокна, синхронного с кадром (Сцены 1, 2, 3)."""

    speech_detected: bool = Field(default=False, description="Обнаружена ли речь (VAD)")
    rms_energy: float = Field(default=0.0, ge=0.0, description="Средняя громкость окна")
    is_loud_spike: bool = Field(
        default=False, description="Резкий акустический пик (крик, удар по стойке)"
    )
    audio_emotion: Emotion = Field(
        default=Emotion.UNKNOWN, description="Распознанная эмоция в голосе"
    )
    audio_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Уверенность аудио-классификатора"
    )


class TrackAffectState(BaseModel):
    """Агрегированное состояние трека после слияния модальностей во временном окне."""

    track_id: int
    final_emotion: Emotion = Field(default=Emotion.UNKNOWN)
    negative_score: float = Field(default=0.0, ge=0.0, le=1.0)
    dominant_modality: str = Field(default="face", description="face | body | audio")
    is_negative_alert: bool = Field(default=False)


class FrameResult(BaseModel):
    """Результат обработки одного кадра — единица обмена между моделью и приложением."""

    frame_id: int = Field(ge=0, description="Порядковый номер кадра")
    detections: List[FaceDetection] = Field(default_factory=list)
    bodies: List[BodyDetection] = Field(default_factory=list)
    audio: Optional[AudioSegmentResult] = Field(default=None)
    track_events: List[TrackAffectState] = Field(default_factory=list)