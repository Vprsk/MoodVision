"""Конфигурация видеопотока и бизнес-логика статусов.

Здесь описаны: источник видеопотока (RTSP/файл/webcam), параметры
обработки и правила агрегации эмоций в понятные пользователю статусы
(Норма / Не уверена / Инцидент).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from contracts.schemas import Emotion


# --- Источники видеопотоков ------------------------------------------------

SOURCE_WEBCAM_PREFIX = "webcam:"


def normalize_source(raw: str) -> str:
    """Приводит пользовательский ввод к строке, понятной cv2.VideoCapture.

    - "webcam:0", "0"      -> "0"      (локальная камера по индексу)
    - "rtsp://...", URL    -> как есть
    - путь к файлу         -> как есть
    """
    raw = raw.strip()
    if raw.lower().startswith(SOURCE_WEBCAM_PREFIX):
        raw = raw[len(SOURCE_WEBCAM_PREFIX):].strip()
    # Веб-камера: оставляем только числовой индекс
    if raw.isdigit():
        return raw
    return raw


# --- Статусы клиентов / эпизодов -------------------------------------------

STATUS_NORMAL = "Норма"
STATUS_UNSURE = "Не уверена"
STATUS_INCIDENT = "Инцидент"

# Эмоции, которые считаем негативными (потенциальный инцидент).
NEGATIVE_EMOTIONS = {
    Emotion.ANGRY,
    Emotion.FEAR,
    Emotion.DISGUST,
    Emotion.SAD,
}
# Эмоции, которые считаем нейтрально-позитивными (норма).
POSITIVE_EMOTIONS = {Emotion.NEUTRAL, Emotion.HAPPY, Emotion.SURPRISE}

# Порог уверенности: ниже — статус «Не уверена».
UNSURE_CONFIDENCE_THRESHOLD = 0.45
# Порог уверенности негативной эмоции для признания инцидента.
INCIDENT_CONFIDENCE_THRESHOLD = 0.60


def classify_status(emotion: Emotion, confidence: float) -> str:
    """Определяет статус по доминирующей эмоции и уверенности.

    - негатив + высокая уверенность      -> Инцидент
    - низкая уверенность или unknown     -> Не уверена
    - иначе                              -> Норма
    """
    if emotion == Emotion.UNKNOWN or confidence < UNSURE_CONFIDENCE_THRESHOLD:
        return STATUS_UNSURE
    if emotion in NEGATIVE_EMOTIONS and confidence >= INCIDENT_CONFIDENCE_THRESHOLD:
        return STATUS_INCIDENT
    return STATUS_NORMAL


# --- Конфигурация одного потока --------------------------------------------

@dataclass
class StreamConfig:
    """Полная конфигурация одного видеопотока."""

    name: str
    source: str
    location: str = ""
    pipeline_key: str = "yolo_vit"
    pipeline_params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    stream_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    # UI-уровневые настройки камеры (страница «Настройки → Камеры»):
    # частота анализа, кроп, поворот кадра и т.п. НЕ передаются в модель —
    # хранятся только для интерфейса и могут быть переиспользованы в будущем.
    ui_settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def capture_source(self) -> str:
        """Строка источника, приведённая к формату cv2.VideoCapture."""
        return normalize_source(self.source)

    @property
    def is_live(self) -> bool:
        """True для прямых эфиров (RTSP/HTTP) и веб-камеры.

        У таких источников нет ни длительности, ни возможности перемотки —
        поведение плеера отличается от файла.
        """
        src = self.capture_source
        if src.isdigit():  # веб-камера по индексу
            return True
        lowered = src.lower()
        return lowered.startswith(("rtsp://", "http://", "https://", "rtmp://"))

    @property
    def is_seekable(self) -> bool:
        """True, если источник поддерживает перемотку (только файлы)."""
        return not self.is_live

    @classmethod
    def create(
        cls,
        name: str,
        source: str,
        location: str = "",
        pipeline_key: str = "yolo_vit",
        pipeline_params: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> "StreamConfig":
        return cls(
            name=name.strip() or "Поток",
            source=source.strip(),
            location=location.strip(),
            pipeline_key=pipeline_key,
            pipeline_params=dict(pipeline_params or {}),
            enabled=enabled,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует конфиг в dict для сохранения (JSON)."""
        return {
            "stream_id": self.stream_id,
            "name": self.name,
            "source": self.source,
            "location": self.location,
            "pipeline_key": self.pipeline_key,
            "pipeline_params": dict(self.pipeline_params or {}),
            "enabled": self.enabled,
            "ui_settings": dict(self.ui_settings or {}),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamConfig":
        """Восстанавливает конфиг из dict (устойчив к отсутствующим полям)."""
        stream_id = str(data.get("stream_id") or "").strip()
        return cls(
            name=str(data.get("name", "Поток")),
            source=str(data.get("source", "")),
            location=str(data.get("location", "")),
            pipeline_key=str(data.get("pipeline_key", "yolo_vit")),
            pipeline_params=dict(data.get("pipeline_params") or {}),
            enabled=bool(data.get("enabled", True)),
            stream_id=stream_id or uuid.uuid4().hex[:8],
            ui_settings=dict(data.get("ui_settings") or {}),
        )



@dataclass
class ClientTrack:
    """Агрегированное состояние одного отслеживаемого клиента (трека).

    Копит историю эмоций и определяет доминирующую эмоцию/уверенность
    для панели «Посетители».
    """

    track_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_emotion: Emotion = Emotion.UNKNOWN
    last_confidence: float = 0.0
    # Взвешенная сумма уверенностей по каждой эмоции.
    emotion_weight: Dict[str, float] = field(default_factory=dict)

    def update(self, emotion: Emotion, confidence: float) -> None:
        """Обновляет состояние трека новым наблюдением."""
        self.last_seen = time.time()
        self.last_emotion = emotion
        self.last_confidence = confidence
        key = emotion.value
        self.emotion_weight[key] = self.emotion_weight.get(key, 0.0) + confidence

    @property
    def dominant_emotion(self) -> Emotion:
        """Эмоция с максимальной накопленной уверенностью."""
        if not self.emotion_weight:
            return Emotion.UNKNOWN
        best = max(self.emotion_weight.items(), key=lambda kv: kv[1])[0]
        try:
            return Emotion(best)
        except ValueError:
            return Emotion.UNKNOWN

    @property
    def dominant_confidence(self) -> float:
        """Нормализованная уверенность доминирующей эмоции (0..1)."""
        total = sum(self.emotion_weight.values())
        if total <= 0.0:
            return self.last_confidence
        best = self.emotion_weight.get(self.dominant_emotion.value, 0.0)
        # Нормируем накопленный вес, но не даём утонуть в длинной истории.
        return min(1.0, best / total * min(total, 1.0) + self.last_confidence * 0.5)

    @property
    def status(self) -> str:
        return classify_status(self.dominant_emotion, self.dominant_confidence)
