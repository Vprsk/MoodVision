"""Миниатюра видеопотока для карточки в списке камер.

Виджет показывает последний кадр потока в небольшом прямоугольнике со
скруглёнными углами. Принимает кадр в формате BGR (как отдаёт OpenCV),
конвертирует его в QPixmap и масштабирует с сохранением пропорций.

Если поток офлайн (остановлен/нет сигнала), последний кадр затемняется, а
поверх выводится подпись — так видно, что картинка «застывшая».
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from app import theme as T

# Радиус скругления углов миниатюры (px).
CORNER_RADIUS = 8


class Thumbnail(QWidget):
    """Показывает последний кадр потока в компактном виде."""

    def __init__(
        self,
        size: Tuple[int, int] = (92, 54),
        placeholder: str = "Нет сигнала",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._placeholder = placeholder
        self._offline = False

        self.setFixedSize(*size)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # Сбрасываем унаследованные от карточки QSS-правила (рамку/фон) —
        # свою рамку миниатюра рисует сама, со скруглёнными углами.
        self.setStyleSheet("background: transparent; border: none;")
        # Клики по миниатюре должны доходить до карточки потока (выбор камеры).
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setToolTip("Миниатюра потока")

    # --- Обновление содержимого ---------------------------------------------

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """Принимает кадр BGR и показывает его в миниатюре."""
        if frame_bgr is None:
            return
        try:
            size = getattr(frame_bgr, "size", 0)
        except Exception:  # noqa: BLE001 — защита от мусора в сигнале
            return
        if not size:
            return
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
        except Exception:  # noqa: BLE001 — неверный формат кадра
            return
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        # .copy() — QImage не должен ссылаться на перезаписываемый буфер numpy.
        self._pixmap = QPixmap.fromImage(image.copy())
        self.update()

    def clear(self) -> None:
        """Убирает кадр — остаётся только заглушка."""
        self._pixmap = None
        self.update()

    def set_offline(self, offline: bool) -> None:
        """Включает/выключает «застывший» вид (поток остановлен)."""
        self._offline = offline
        self.update()

    # --- Отрисовка -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, CORNER_RADIUS, CORNER_RADIUS)

        painter.setClipPath(path)
        painter.fillRect(self.rect(), QColor(T.BG_DARKEST))

        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            if self._offline:
                # Затемняем «застывший» кадр.
                painter.fillRect(self.rect(), QColor(0, 0, 0, 130))
        else:
            painter.setPen(QColor(T.TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)

        painter.setClipping(False)

        if self._offline and self._pixmap is not None:
            painter.setPen(QColor(T.STATUS_IDLE))
            painter.drawText(
                self.rect().adjusted(0, 0, 0, -2),
                Qt.AlignHCenter | Qt.AlignBottom,
                self._placeholder,
            )

        # Рамка поверх картинки.
        painter.setPen(QPen(QColor(T.BORDER), 1))
        painter.drawPath(path)
        painter.end()