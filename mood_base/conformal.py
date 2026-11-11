"""Конформный контроль ложных тревог (Conformal Prediction).

Идея: вместо жёсткого фиксированного порога уверенности (например, 70%),
мы калибруем порог на отдельной выборке так, чтобы гарантировать целевой
уровень FDR (False Discovery Rate). Когда модель недостаточно уверена,
система возвращает «abstain» («не знаю», молчание).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from contracts.schemas import Emotion

NEGATIVE_EMOTION_VALUES = {
    Emotion.ANGRY.value,
    Emotion.SAD.value,
    Emotion.FEAR.value,
    Emotion.DISGUST.value,
}


class ConformalController:
    """Конформный контроллер для принятия решений о тревоге.

    Калибрует порог уверенности на выборке с известной разметкой, чтобы
    обеспечить целевой уровень FDR на новых данных.
    """

    def __init__(self, target_fdr: float = 0.10) -> None:
        self.target_fdr = target_fdr
        self.threshold: float | None = None
        self._calibrated = False

    def calibrate(
        self,
        scores: List[float],
        is_incident: List[bool],
    ) -> float:
        """Калибрует порог уверенности.

        Args:
            scores: уверенность модели (0..1) для каждого примера
            is_incident: True если это реальный инцидент (негативная эмоция)

        Returns:
            Выбранный порог уверенности.
        """
        scores_arr = np.asarray(scores, dtype=np.float32)
        incident_arr = np.asarray(is_incident, dtype=bool)

        # Сортируем по убыванию confidence
        order = np.argsort(-scores_arr)
        sorted_scores = scores_arr[order]
        sorted_incident = incident_arr[order]

        # Вычисляем накопительный FDR на каждом пороге
        tp_cum = np.cumsum(sorted_incident)
        fp_cum = np.cumsum(~sorted_incident)
        total_cum = tp_cum + fp_cum
        fdr_cum = np.where(total_cum > 0, fp_cum / np.maximum(total_cum, 1), 0.0)

        # Находим самый высокий порог, при котором FDR <= целевой
        valid = fdr_cum <= self.target_fdr
        if np.any(valid):
            # Берём последний валидный порог (самый высокий confidence)
            idx = np.max(np.where(valid)[0])
            self.threshold = float(sorted_scores[idx])
        else:
            # Не можем достичь целевого FDR — ставим порог 1.0 (всегда abstain)
            self.threshold = 1.0

        self._calibrated = True
        return self.threshold

    def predict(
        self,
        emotion: str,
        confidence: float,
    ) -> Tuple[str, float, bool]:
        """Принимает решение о тревоге.

        Returns:
            (решение, confidence, is_alert)
            решение: эмоция или 'abstain' если ниже порога
            is_alert: True если это тревога (негативная эмоция + выше порога)
        """
        if not self._calibrated:
            raise RuntimeError("ConformalController не откалиброван. Вызовите calibrate() primero.")

        if self.threshold is None or confidence < self.threshold:
            return "abstain", confidence, False

        is_alert = emotion in NEGATIVE_EMOTION_VALUES
        return emotion, confidence, is_alert
