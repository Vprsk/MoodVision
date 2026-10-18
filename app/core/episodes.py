"""Вычисление эпизодов и сводной статистики из лога анализа файла.

Лог (log.json) содержит кадры с детекциями. Эпизод — непрерывный отрезок
времени, в течение которого один трек (клиент) присутствует в кадре.
Для каждого эпизода определяется доминирующая эмоция, доля негатива и
статус (Норма / Не уверена / Инцидент) — теми же правилами, что и для
live-потоков (app.core.stream_config.classify_status).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.core.stream_config import (
    NEGATIVE_EMOTIONS,
    STATUS_INCIDENT,
    STATUS_NORMAL,
    STATUS_UNSURE,
    classify_status,
)
from contracts.schemas import Emotion

# Разрыв между наблюдениями трека (в секундах), при котором эпизод считается
# прерванным и начинается новый.
EPISODE_GAP_SECONDS = 3.0

# Эмоции на русском (для UI).
EMOTION_RU: Dict[str, str] = {
    "neutral": "Нейтрально",
    "happy": "Радость",
    "sad": "Грусть",
    "angry": "Злость",
    "surprise": "Удивление",
    "fear": "Страх",
    "disgust": "Отвращение",
    "unknown": "Не определено",
}

# Негативные эмоции в формате строк значений Emotion.
_NEGATIVE_VALUES = {e.value for e in NEGATIVE_EMOTIONS}


@dataclass
class Episode:
    """Один эпизод присутствия клиента в кадре."""

    track_id: int
    start_sec: float
    end_sec: float
    dominant_emotion: str = "unknown"
    negative: float = 0.0        # доля негативных наблюдений 0..1
    status: str = STATUS_NORMAL
    observations: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def time_range(self) -> str:
        """Отрезок времени в формате «ММ:СС–ММ:СС»."""

        def _fmt(sec: float) -> str:
            sec = max(0, int(sec))
            m, s = divmod(sec, 60)
            return f"{m:02d}:{s:02d}"

        return f"{_fmt(self.start_sec)}–{_fmt(self.end_sec)}"


@dataclass
class AnalysisStats:
    """Сводная статистика по всей записи."""

    episodes: List[Episode] = field(default_factory=list)
    incidents: int = 0
    unsure: int = 0
    normal: int = 0
    suitable_percent: float = 0.0   # доля кадров с лицами от всех кадров
    duration_sec: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0


def compute_episodes(log: Dict[str, Any]) -> AnalysisStats:
    """Строит эпизоды и метрики по JSON-логу анализа (log.json).

    Каждая детекция с track_id привязывается к эпизоду своего трека; наблюдения
    с разрывом > EPISODE_GAP_SECONDS начинают новый эпизод. Статус эпизода
    определяется доминирующей эмоцией и средней уверенностью (classify_status).
    """
    stats = AnalysisStats()
    video = log.get("video") or {}
    stats.fps = float(video.get("fps") or 0.0)
    stats.duration_sec = float(video.get("duration_sec") or 0.0)
    stats.width = int(video.get("width") or 0)
    stats.height = int(video.get("height") or 0)

    total_frames = int(video.get("total_frames") or 0)
    results = log.get("results") or []

    # Незакрытые эпизоды по трекам: track_id -> состояние.
    open_eps: Dict[int, Dict[str, Any]] = {}
    frames_with_faces = 0

    for frame in results:
        detections = frame.get("detections") or []
        if not detections:
            continue
        frames_with_faces += 1
        frame_id = int(frame.get("frame_id") or 0)
        t = frame_id / stats.fps if stats.fps > 0 else float(frame_id)

        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                continue
            emotion = det.get("emotion") or "unknown"
            conf = float(det.get("emotion_confidence") or 0.0)

            state = open_eps.get(tid)
            if state is not None and t - state["end"] > EPISODE_GAP_SECONDS:
                # Разрыв слишком большой — закрываем предыдущий эпизод.
                stats.episodes.append(_close_episode(tid, state))
                state = None
            if state is None:
                state = {"start": t, "end": t, "weights": {}, "obs": 0}
                open_eps[tid] = state
            state["end"] = t
            state["obs"] += 1
            state["weights"][emotion] = state["weights"].get(emotion, 0.0) + max(
                conf, 0.01
            )

    # Закрываем оставшиеся эпизоды.
    for tid, state in open_eps.items():
        stats.episodes.append(_close_episode(tid, state))

    stats.episodes.sort(key=lambda ep: ep.start_sec)

    for ep in stats.episodes:
        if ep.status == STATUS_INCIDENT:
            stats.incidents += 1
        elif ep.status == STATUS_UNSURE:
            stats.unsure += 1
        else:
            stats.normal += 1

    if total_frames > 0:
        stats.suitable_percent = round(frames_with_faces / total_frames * 100.0, 1)
    elif results:
        stats.suitable_percent = 100.0

    # Длительность — из видео; если её нет, берём последний эпизод.
    if stats.duration_sec <= 0 and stats.episodes:
        stats.duration_sec = stats.episodes[-1].end_sec

    return stats


def _close_episode(track_id: int, state: Dict[str, Any]) -> Episode:
    """Собирает Episode из накопленного состояния трека."""
    weights: Dict[str, float] = state["weights"]
    total = sum(weights.values()) or 1.0
    dominant = max(weights.items(), key=lambda kv: kv[1])[0] if weights else "unknown"

    # Доля негативных наблюдений.
    negative_weight = sum(w for e, w in weights.items() if e in _NEGATIVE_VALUES)
    negative = min(1.0, negative_weight / total)

    # Уверенность доминирующей эмоции (нормализованная).
    confidence = weights.get(dominant, 0.0) / total

    try:
        status = classify_status(Emotion(dominant), confidence)
    except ValueError:
        # Некорректная строка эмоции в логе — считаем «Не уверена».
        status = STATUS_UNSURE

    return Episode(
        track_id=track_id,
        start_sec=state["start"],
        end_sec=state["end"],
        dominant_emotion=dominant,
        negative=round(negative, 2),
        status=status,
        observations=state["obs"],
    )