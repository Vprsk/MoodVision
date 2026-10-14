"""Отрисовка результатов детекции на кадре.

Логика перенесена из scripts/try.py и адаптирована: работает с любым
FrameResult контракта, ничего не знает о конкретных моделях.
"""

from __future__ import annotations

import cv2
import numpy as np

from contracts.schemas import Emotion, FaceDetection

# Цвет рамки по эмоции (BGR). Совпадает с палитрой прототипа try.py.
EMOTION_COLORS: dict[Emotion, tuple[int, int, int]] = {
    Emotion.UNKNOWN: (0, 165, 255),
    Emotion.HAPPY: (0, 255, 255),
    Emotion.SAD: (255, 0, 0),
    Emotion.ANGRY: (0, 0, 255),
    Emotion.SURPRISE: (255, 0, 255),
    Emotion.FEAR: (128, 0, 128),
    Emotion.DISGUST: (0, 128, 0),
    Emotion.NEUTRAL: (200, 200, 200),
}
DEFAULT_COLOR = (0, 255, 0)


def color_for_emotion(emotion: Emotion) -> tuple[int, int, int]:
    """Возвращает цвет (BGR) для рамки по эмоции."""
    return EMOTION_COLORS.get(emotion, DEFAULT_COLOR)


def draw_detection(frame: np.ndarray, det: FaceDetection) -> np.ndarray:
    """Рисует одну детекцию (bbox + подпись) на переданном кадре (in-place)."""
    x1, y1 = det.bbox.x_min, det.bbox.y_min
    x2, y2 = det.bbox.x_max, det.bbox.y_max
    color = color_for_emotion(det.emotion)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    track = f"#{det.track_id}" if det.track_id is not None else "#?"
    label = f"{track} {det.emotion.value} {det.emotion_confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(15, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Рисует счётчик FPS в левом верхнем углу."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame
