"""Baseline: простое голосование эмоций в окне 1.5 секунды.

Ждёт накопления наблюдений в течение окна времени (по умолчанию 1.5 с),
затем выбирает самую частую эмоцию (взвешенную по уверенности) как предсказание.
Если окно не заполнено или лицо не найдено — unknown.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple


class MajorityVoteBaseline:
    """Базовый метод: majority vote во временном окне."""

    def __init__(
        self,
        window_seconds: float = 1.5,
        min_confidence: float = 0.0,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_confidence = min_confidence

        # track_id -> deque of (timestamp, emotion, confidence)
        self._observations: Dict[int, Deque[Tuple[float, str, float]]] = defaultdict(
            lambda: deque()
        )
        # Финальные решения
        self._decisions: Dict[int, Tuple[str, float]] = {}
        # Флаг: был ли трек уже обработан (решение принято)
        self._finalized: Dict[int, bool] = defaultdict(bool)

    def update(
        self,
        track_id: int,
        timestamp: float,
        emotion: str,
        confidence: float,
    ) -> None:
        """Добавляет наблюдение для трека."""
        if self._finalized.get(track_id, False):
            return

        self._observations[track_id].append((timestamp, emotion, confidence))

        # Удаляем наблюдения старше окна
        cutoff = timestamp - self.window_seconds
        obs = self._observations[track_id]
        while obs and obs[0][0] < cutoff:
            obs.popleft()

    def _compute_decision(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Вычисляет решение на основе текущих наблюдений в окне."""
        obs = self._observations.get(track_id)
        if not obs:
            return None

        # Взвешенное голосование по уверенности
        emotion_weights: Dict[str, float] = defaultdict(float)
        total_weight = 0.0

        for _, emotion, confidence in obs:
            if confidence < self.min_confidence:
                continue
            emotion_weights[emotion] += confidence
            total_weight += confidence

        if total_weight == 0:
            return None

        best_emotion = max(emotion_weights, key=emotion_weights.get)
        best_conf = emotion_weights[best_emotion] / total_weight
        return (best_emotion, min(best_conf, 1.0))

    def decision(self, track_id: int, current_timestamp: float) -> Optional[Tuple[str, float]]:
        """Возвращает решение для трека.

        Если уже принято — возвращает кэшированное.
        Иначе пытается принять решение на основе наблюдений в окне.
        """
        if track_id in self._decisions:
            return self._decisions[track_id]

        if self._finalized.get(track_id, False):
            return self._decisions.get(track_id)

        dec = self._compute_decision(track_id)
        if dec is not None:
            self._decisions[track_id] = dec
            self._finalized[track_id] = True

        return dec

    def finalize(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Принудительно принимает решение для трека (например, в конце видео).

        Использует все накопленные наблюдения, не ограниченные окном.
        """
        if track_id in self._decisions:
            return self._decisions[track_id]

        obs = self._observations.get(track_id)
        if not obs:
            return None

        emotion_weights: Dict[str, float] = defaultdict(float)
        total_weight = 0.0

        for _, emotion, confidence in obs:
            if confidence < self.min_confidence:
                continue
            emotion_weights[emotion] += confidence
            total_weight += confidence

        if total_weight == 0:
            self._decisions[track_id] = ("unknown", 0.0)
            self._finalized[track_id] = True
            return self._decisions[track_id]

        best_emotion = max(emotion_weights, key=emotion_weights.get)
        best_conf = emotion_weights[best_emotion] / total_weight
        self._decisions[track_id] = (best_emotion, min(best_conf, 1.0))
        self._finalized[track_id] = True
        return self._decisions[track_id]

    def reset_track(self, track_id: int) -> None:
        self._observations.pop(track_id, None)
        self._decisions.pop(track_id, None)
        self._finalized.pop(track_id, None)
