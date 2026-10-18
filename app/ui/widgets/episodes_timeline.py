"""Таймлайн эпизодов записи: горизонтальная полоса с отрезками эпизодов.

По референсу: полоса во всю ширину, отрезки окрашены по статусу
(Инцидент — оранжевый, Не уверена — серо-синий, Норма — тёмный),
под полосой — тайм-метки начала/конца и промежуточных точек.
"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from app import theme as T
from app.core.episodes import Episode
from app.core.stream_config import STATUS_INCIDENT, STATUS_UNSURE

# Высота полосы в пикселях.
_BAR_HEIGHT = 44
# Отступ снизу под тайм-метки.
_LABELS_HEIGHT = 22

_STATUS_COLORS = {
    STATUS_INCIDENT: T.STATUS_INCIDENT,
    STATUS_UNSURE: "#4b5a7d",
}


def status_color(status: str) -> str:
    """Цвет сегмента таймлайна по статусу эпизода."""
    return _STATUS_COLORS.get(status, "#1d2a45")


class EpisodesTimeline(QWidget):
    """Полоса эпизодов с легендой и тайм-метками."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._episodes: List[Episode] = []
        self._duration: float = 0.0
        self.setMinimumHeight(_BAR_HEIGHT + _LABELS_HEIGHT + 8)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, episodes: List[Episode], duration_sec: float) -> None:
        """Устанавливает эпизоды и общую длительность записи."""
        self._episodes = list(episodes)
        self._duration = max(duration_sec, 1.0)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        bar_h = _BAR_HEIGHT

        # Подложка полосы.
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(T.BG_DARKEST))
        painter.drawRoundedRect(0, 0, width, bar_h, 6, 6)

        # Сегменты эпизодов.
        for ep in self._episodes:
            x1 = ep.start_sec / self._duration * width
            x2 = ep.end_sec / self._duration * width
            if x2 - x1 < 4:
                x2 = x1 + 4  # минимальная видимая ширина сегмента
            color = QColor(status_color(ep.status))
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRect(int(x1), 4, int(x2 - x1), bar_h - 8)

        # Тайм-метки под полосой.
        pen = QPen(QColor(T.TEXT_MUTED))
        painter.setPen(pen)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        marks = 5
        for i in range(marks):
            ratio = i / (marks - 1)
            sec = ratio * self._duration
            m, s = divmod(int(sec), 60)
            label = f"{m:02d}:{s:02d}"
            x = ratio * width
            if i == 0:
                align = Qt.AlignLeft
            elif i == marks - 1:
                align = Qt.AlignRight
            else:
                align = Qt.AlignHCenter
            painter.drawText(
                int(x) - (60 if align == Qt.AlignHCenter else 0),
                bar_h + 4,
                120 if align == Qt.AlignHCenter else width,
                _LABELS_HEIGHT,
                align | Qt.AlignVCenter,
                label,
            )

        painter.end()


class TimelineCard(QWidget):
    """Карточка «Эпизоды на записи»: заголовок + легенда + таймлайн."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setStyleSheet(
            f"background: {T.BG_CARD}; border: 1px solid {T.BORDER};"
            " border-radius: 12px;"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        # --- Шапка: заголовок + легенда ---
        header = QWidget()
        header.setStyleSheet("background: transparent; border: none;")
        hl = QVBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)

        title = QLabel("Эпизоды на записи")
        title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 15px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        hl.addWidget(title)

        legend = QWidget()
        legend.setStyleSheet("background: transparent; border: none;")
        ll = QHBoxLayout(legend)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(16)
        for text, color in (
            ("Инцидент", T.STATUS_INCIDENT),
            ("Не уверена", "#4b5a7d"),
            ("Нет лиц в кадре", T.BG_DARKEST),
        ):
            dot = QLabel("■")
            dot.setStyleSheet(
                f"color: {color}; background: transparent; border: none;"
                " font-size: 12px;"
            )
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {T.TEXT_SECONDARY}; font-size: 12px;"
                " background: transparent; border: none;"
            )
            ll.addWidget(dot)
            ll.addWidget(lbl)
        ll.addStretch(1)
        hl.addWidget(legend)

        root.addWidget(header)

        self.timeline = EpisodesTimeline()
        root.addWidget(self.timeline)

    def set_data(self, episodes: List[Episode], duration_sec: float) -> None:
        self.timeline.set_data(episodes, duration_sec)