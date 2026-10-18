"""Переключатели-сегменты и тумблер для страницы настроек камер.

SegmentedControl — горизонтальный ряд взаимоисключающих кнопок (как в
референсе: «Авто | 384 | 640 | …»). ToggleSwitch — анимированный тумблер.
Оба виджета возвращают выбранное значение строкой и эмитят valueChanged.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from app import theme as T


class SegmentedControl(QWidget):
    """Ряд взаимоисключающих кнопок-сегментов."""

    valueChanged = Signal(str)

    def __init__(
        self, options: List[str], current: Optional[str] = None, parent=None
    ) -> None:
        super().__init__(parent)
        self._options = list(options)
        self._current = current if current in self._options else (
            self._options[0] if self._options else ""
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        for option in self._options:
            btn = QPushButton(option)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setObjectName("SegmentButton")
            btn.setFixedHeight(32)
            btn.clicked.connect(lambda _=False, o=option: self._on_click(o))
            self._group.addButton(btn)
            self._buttons[option] = btn
            layout.addWidget(btn)

        self._apply_checked()

    def _on_click(self, option: str) -> None:
        if option == self._current:
            return
        self._current = option
        self._apply_checked()
        self.valueChanged.emit(option)

    def _apply_checked(self) -> None:
        for option, btn in self._buttons.items():
            btn.setChecked(option == self._current)

    def value(self) -> str:
        return self._current

    def set_value(self, value: str) -> None:
        if value in self._buttons and value != self._current:
            self._current = value
            self._apply_checked()


class ToggleSwitch(QWidget):
    """Анимированный тумблер (вкл/выкл)."""

    valueChanged = Signal(bool)

    def __init__(self, checked: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._checked = checked
        self._pos = 1.0 if checked else 0.0
        self.setFixedSize(44, 24)
        self.setCursor(Qt.PointingHandCursor)

        self._anim = QPropertyAnimation(self, b"pos_ratio", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)

    def get_pos_ratio(self) -> float:
        return self._pos

    def set_pos_ratio(self, value: float) -> None:
        self._pos = value
        self.update()

    pos_ratio = Property(float, get_pos_ratio, set_pos_ratio)

    def isChecked(self) -> bool:  # noqa: N802 (Qt API)
        return self._checked

    def set_checked(self, checked: bool, animate: bool = True) -> None:
        if checked == self._checked:
            return
        self._checked = checked
        target = 1.0 if checked else 0.0
        if animate:
            self._anim.stop()
            self._anim.setStartValue(self._pos)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._pos = target
            self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.button() == Qt.LeftButton:
            self.set_checked(not self._checked)
            self.valueChanged.emit(self._checked)
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        track = QColor(T.ACCENT if self._checked else T.BORDER_STRONG)
        painter.setPen(Qt.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 12, 12)

        r = self.height() - 6
        x = 3 + (self.width() - 6 - r) * self._pos
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(int(x), 3, r, r, r // 2, r // 2)
        painter.end()