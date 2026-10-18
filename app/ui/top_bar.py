"""Верхняя панель приложения: логотип, меню-разделы.

Меню реализовано как набор взаимно-исключающих кнопок-переключателей,
сигнализирующих индекс открываемой страницы в QStackedWidget. Пункты и
порядок соответствуют референсу: Потоки / История / Настройки / Анализ видео.
"""

from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from app import theme as T

# (Ключ, подпись) пунктов меню.
MENU_ITEMS: List[Tuple[str, str]] = [
    ("streams", "Потоки"),
    ("history", "История"),
    ("settings", "Настройки"),
    ("files", "Анализ видео"),
]


class TopBar(QWidget):
    """Панель навигации вверху окна."""

    navigate = Signal(str)  # ключ раздела

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(58)

        self._buttons: dict[str, QPushButton] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setSpacing(6)

        # --- Логотип ---
        logo = QLabel("☻")
        logo.setStyleSheet(f"color: {T.ACCENT}; font-size: 22px;")
        title = QLabel("EmoSense")
        title.setObjectName("AppTitle")
        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addSpacing(28)

        # --- Пункты меню ---
        for key, label in MENU_ITEMS:
            btn = QPushButton(f"{label}  ▾")
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.clicked.connect(lambda checked=False, k=key: self.navigate.emit(k))
            self._buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch(1)

        # Активный раздел по умолчанию.
        self.set_active("streams")

    def set_active(self, key: str) -> None:
        """Отмечает активный раздел (подсвечивает кнопку)."""
        for k, btn in self._buttons.items():
            btn.setChecked(k == key)
