"""Виджет для отображения видеокадра (BGR numpy -> QImage).

Сохраняет пропорции кадра при масштабировании под размер виджета и
показывает заглушку, когда кадра ещё нет.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from app import theme as T


class VideoSurface(QWidget):
    """Показывает последний полученный кадр, масштабируя его с сохранением пропорций."""

    def __init__(self, placeholder: str = "Нет сигнала", parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 200)
        self.setStyleSheet(f"background: {T.BG_DARKEST}; border-radius: 10px;")

        self._placeholder = placeholder
        self._pixmap: Optional[QPixmap] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(placeholder, self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(f"color: {T.TEXT_MUTED}; background: transparent;")
        self._label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._label.setScaledContents(False)
        layout.addWidget(self._label)

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Принимает кадр в формате BGR (как отдаёт OpenCV) и отображает его."""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # .copy() — чтобы QImage не ссылался на буфер numpy, который перезапишется.
        self._pixmap = QPixmap.fromImage(image.copy())
        self._label.setText("")
        self._refresh_pixmap()

    def set_placeholder(self, text: str) -> None:
        """Показывает текстовую заглушку вместо кадра."""
        self._placeholder = text
        self._pixmap = None
        self._label.setText(text)
        self._label.setPixmap(QPixmap())

    def sizeHint(self) -> QSize:
        return QSize(480, 300)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        """Масштабирует текущий кадр под размер виджета с сохранением пропорций."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        target = self._label.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._pixmap.scaled(
            target,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
