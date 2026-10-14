"""Баннер инцидента вверху страницы «Потоки».

Показывает предупреждение о негативном клиенте с кнопками действий
(Подтвердить / Ложное / Скрыть).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from app import theme as T


class IncidentBanner(QFrame):
    """Оранжевая полоса-предупреждение о зафиксированном инциденте."""

    confirm = Signal()
    dismiss = Signal()
    mark_false = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._normal_style = (
            f"background: {T.BG_CARD}; border: 1px solid {T.BORDER};"
            " border-radius: 10px;"
        )
        self._alert_style = (
            f"background: {T.soft(T.STATUS_INCIDENT, 0.12)};"
            f" border: 1px solid {T.STATUS_INCIDENT}; border-radius: 10px;"
        )
        self.setStyleSheet(self._normal_style)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        warn = QLabel("⚠")
        warn.setStyleSheet(f"font-size: 18px; color: {T.STATUS_INCIDENT};")
        layout.addWidget(warn)

        self._text = QLabel("Инцидентов нет")
        self._text.setStyleSheet(f"color: {T.TEXT_PRIMARY}; font-size: 14px;")
        layout.addWidget(self._text, 1)

        self._confirm_btn = QPushButton("Подтвердить")
        self._confirm_btn.clicked.connect(self.confirm.emit)
        self._false_btn = QPushButton("Ложное")
        self._false_btn.clicked.connect(self.mark_false.emit)
        self._hide_btn = QPushButton("Скрыть")
        self._hide_btn.setStyleSheet(
            f"background: transparent; border: none; color: {T.TEXT_SECONDARY};"
        )
        self._hide_btn.clicked.connect(self.dismiss.emit)

        layout.addWidget(self._confirm_btn)
        layout.addWidget(self._false_btn)
        layout.addWidget(self._hide_btn)

        self.set_incident(None)

    def set_incident(self, message: str | None) -> None:
        """Показывает баннер с сообщением либо скрывает содержимое."""
        has = message is not None
        self._text.setText(message or "Инцидентов нет")
        self._confirm_btn.setVisible(has)
        self._false_btn.setVisible(has)
        self._hide_btn.setVisible(has)
        self.setStyleSheet(self._alert_style if has else self._normal_style)
