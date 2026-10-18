"""Небольшие переиспользуемые виджеты: бейдж-метка и карточка-метрика."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from app import theme as T


class Badge(QLabel):
    """Цветная метка (статус, эмоция) с обводкой нужного цвета."""

    def __init__(self, text: str, color: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self._apply_color(color)

    def _apply_color(self, color: str) -> None:
        self.setStyleSheet(
            f"""
            color: {color};
            border: 1px solid {color};
            border-radius: 10px;
            padding: 2px 10px;
            font-size: 12px;
            font-weight: 600;
            """
        )

    def set_badge(self, text: str, color: str) -> None:
        """Обновляет текст и цвет бейджа."""
        self.setText(text)
        self._apply_color(color)


class MetricCard(QFrame):
    """Карточка с заголовком, крупным значением и подписью (для статистики)."""

    def __init__(self, title: str, value: str = "—", caption: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)

        self._title = QLabel(title)
        self._title.setStyleSheet(f"color: {T.TEXT_SECONDARY}; font-size: 13px;")
        layout.addWidget(self._title)

        self._value = QLabel(value)
        self._value.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 30px; font-weight: 700;"
        )
        layout.addWidget(self._value)

        self._caption = QLabel(caption)
        self._caption.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(self._caption)

    def set_value(self, value: str, caption: str = "") -> None:
        """Обновляет значение и подпись карточки."""
        self._value.setText(value)
        if caption:
            self._caption.setText(caption)
