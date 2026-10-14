"""Программно нарисованные иконки для UI (без внешних файлов/SVG).

Иконки рисуются векторно через QPainter и рендерятся с высоким device
pixel ratio, поэтому остаются чёткими на любых экранах. Так избегаем
«кривых» юникод-символов вроде «＋».
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap


def make_plus_icon(
    box: int = 24,
    color: str = "#ffffff",
    arm: float = 0.56,        # длина луча (доля от box)
    thickness: float = 0.14,  # толщина луча (доля от box)
    corner: float = 0.5,      # скругление концов (доля от толщины)
    dpr: int = 3,
) -> QIcon:
    """Возвращает QIcon с аккуратно нарисованным «плюсом».

    Плюс собран из двух скруглённых прямоугольников (лучей), пересекающихся
    в центре — визуально как SVG-иконка, с антиалиасингом и чёткими краями.
    """
    px = QPixmap(box * dpr, box * dpr)
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.transparent)

    painter = QPainter(px)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))

    c = box / 2.0
    arm_len = box * arm / 2.0
    thick = box * thickness
    rad = thick * corner

    # Горизонтальный луч
    painter.drawRoundedRect(
        QRectF(c - arm_len, c - thick / 2.0, arm_len * 2.0, thick), rad, rad
    )
    # Вертикальный луч
    painter.drawRoundedRect(
        QRectF(c - thick / 2.0, c - arm_len, thick, arm_len * 2.0), rad, rad
    )
    painter.end()

    return QIcon(px)


def _icon_from_painter(box: int, dpr: int, draw_fn) -> QIcon:
    """Создаёт QIcon, делегируя отрисовку функции draw_fn(painter, box)."""
    px = QPixmap(box * dpr, box * dpr)
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.transparent)

    painter = QPainter(px)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#ffffff"))
    draw_fn(painter, box)
    painter.end()

    return QIcon(px)


def make_play_icon(box: int = 24, dpr: int = 3) -> QIcon:
    """Иконка «Play»: скруглённый треугольник, направленный вправо."""

    def draw(painter: QPainter, b: int) -> None:
        path = QPainterPath()
        # Небольшой сдвиг вправо для визуального центрирования треугольника.
        path.moveTo(b * 0.34, b * 0.24)
        path.lineTo(b * 0.80, b * 0.50)
        path.lineTo(b * 0.34, b * 0.76)
        path.closeSubpath()
        painter.drawPath(path)

    return _icon_from_painter(box, dpr, draw)


def make_pause_icon(box: int = 24, dpr: int = 3) -> QIcon:
    """Иконка «Pause»: две скруглённые вертикальные полоски."""

    def draw(painter: QPainter, b: int) -> None:
        bar_w = b * 0.18
        gap = b * 0.16
        total = bar_w * 2 + gap
        x0 = (b - total) / 2.0
        painter.drawRoundedRect(
            QRectF(x0, b * 0.24, bar_w, b * 0.52), bar_w * 0.35, bar_w * 0.35
        )
        painter.drawRoundedRect(
            QRectF(x0 + bar_w + gap, b * 0.24, bar_w, b * 0.52),
            bar_w * 0.35,
            bar_w * 0.35,
        )

    return _icon_from_painter(box, dpr, draw)


def make_seek_back_icon(box: int = 24, dpr: int = 3) -> QIcon:
    """Иконка «Перемотать назад»: двойной треугольник влево."""

    def draw(painter: QPainter, b: int) -> None:
        # Два треугольника, направленных влево.
        for x_off in (0.52, 0.24):
            path = QPainterPath()
            path.moveTo(b * x_off, b * 0.24)
            path.lineTo(b * x_off, b * 0.76)
            path.lineTo(b * (x_off - 0.26), b * 0.50)
            path.closeSubpath()
            painter.drawPath(path)

    return _icon_from_painter(box, dpr, draw)


def make_seek_forward_icon(box: int = 24, dpr: int = 3) -> QIcon:
    """Иконка «Перемотать вперёд»: двойной треугольник вправо."""

    def draw(painter: QPainter, b: int) -> None:
        for x_off in (0.24, 0.52):
            path = QPainterPath()
            path.moveTo(b * x_off, b * 0.24)
            path.lineTo(b * x_off, b * 0.76)
            path.lineTo(b * (x_off + 0.26), b * 0.50)
            path.closeSubpath()
            painter.drawPath(path)

    return _icon_from_painter(box, dpr, draw)
