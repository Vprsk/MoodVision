"""Sequential Probability Ratio Test (SPRT) для принятия решений об эмоции.

Для каждого трека ведётся последовательное накопление логарифма отношения
правдоподобия (LLR) между гипотезами «эмоция E» vs «не-E».
Когда LLR пересекает верхнюю границу — принимается решение E.
Когда пересекает нижнюю — эмоция E отвергается (трек закрывается как unknown).
Между границами — продолжаем ждать следующий кадр.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from contracts.schemas import Emotion


class SPRTDetector:
    """SPRT-детектор для последовательного принятия решения об эмоции трека."""

    def __init__(
        self,
        alpha: float = 0.05,  # вероятность ложноположительного решения
        beta: float = 0.05,   # вероятность ложноотрицательного решения
        min_confidence: float = 0.1,
        max_frames: int = 100,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.min_confidence = min_confidence
        self.max_frames = max_frames

        # Границы SPRT
        self.upper_boundary = math.log((1.0 - beta) / alpha)
        self.lower_boundary = math.log(beta / (1.0 - alpha))

        # Состояние по трекам: track_id -> {emotion: LLR}
        self._llr: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # История наблюдений по трекам
        self._observations: Dict[int, Deque[Tuple[str, float]]] = defaultdict(lambda: deque(maxlen=max_frames))
        # Количество кадров по трекам
        self._frame_counts: Dict[int, int] = defaultdict(int)
        # Финальные решения по трекам
        self._decisions: Dict[int, Tuple[str, float]] = {}

    def update(self, track_id: int, emotion: str, confidence: float) -> None:
        """Добавляет новое наблюдение для трека."""
        if track_id in self._decisions:
            return  # Решение уже принято

        conf = max(confidence, self.min_confidence)
        self._frame_counts[track_id] += 1
        self._observations[track_id].append((emotion, conf))

        # Обновляем LLR для всех возможных эмоций
        for em in Emotion:
            em_val = em.value
            # P(obs | H1: эмоция = em)
            if emotion == em_val:
                p_h1 = conf
            else:
                p_h1 = 1.0 - conf
            p_h1 = max(p_h1, 1e-6)

            # P(obs | H0: эмоция != em)
            if emotion != em_val:
                p_h0 = conf
            else:
                p_h0 = 1.0 - conf
            p_h0 = max(p_h0, 1e-6)

            llr_increment = math.log(p_h1 / p_h0)
            self._llr[track_id][em_val] += llr_increment

        # Проверяем границы для каждой эмоции
        self._check_boundaries(track_id)

    def _check_boundaries(self, track_id: int) -> None:
        """Проверяет, достигнута ли граница для любой эмоции."""
        for em in Emotion:
            em_val = em.value
            llr = self._llr[track_id].get(em_val, 0.0)

            if llr >= self.upper_boundary:
                # Принимаем эмоцию em
                conf = min(1.0, 1.0 / (1.0 + math.exp(-llr)))
                self._decisions[track_id] = (em_val, conf)
                return
            elif llr <= self.lower_boundary:
                # Отклоняем эмоцию em — убираем из рассмотрения
                # Если все эмоции отклонены, ставим unknown
                pass

    def decision(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Возвращает принятое решение для трека или None, если ещё не принято."""
        if track_id in self._decisions:
            return self._decisions[track_id]

        # Если слишком много кадров и решения нет — принимаем meilleures по LLR
        if self._frame_counts.get(track_id, 0) >= self.max_frames:
            best_em = max(
                self._llr[track_id],
                key=self._llr[track_id].get,
                default="unknown",
            )
            best_llr = self._llr[track_id].get(best_em, 0.0)
            conf = min(1.0, max(0.0, 1.0 / (1.0 + math.exp(-abs(best_llr)))))
            self._decisions[track_id] = (best_em, conf)
            return self._decisions[track_id]

        return None

    def get_current_belief(self, track_id: int) -> Dict[str, float]:
        """Возвращает текущие вероямити beliefs для всех эмоций (softmax от LLR)."""
        if track_id not in self._llr:
            return {}
        llrs = self._llr[track_id]
        max_llr = max(llrs.values()) if llrs else 0.0
        exp_llrs = {em: math.exp(llr - max_llr) for em, llr in llrs.items()}
        total = sum(exp_llrs.values())
        if total == 0:
            return {em: 1.0 / len(exp_llrs) for em in exp_llrs}
        return {em: val / total for em, val in exp_llrs.items()}

    def reset_track(self, track_id: int) -> None:
        """Сбрасывает состояние трека."""
        self._llr.pop(track_id, None)
        self._observations.pop(track_id, None)
        self._frame_counts.pop(track_id, None)
        self._decisions.pop(track_id, None)
