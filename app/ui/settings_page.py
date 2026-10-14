"""Страница «Настройки»: параметры приложения.

Выбор пайплайна и параметров обработки переехал на вкладку «Пресеты» —
здесь отображается краткая справка, какие параметры доступны.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.core.pipeline_registry import get_pipelines
from app.core.stream_manager import StreamManager


class SettingsPage(QWidget):
    """Настройки приложения."""

    def __init__(self, manager: StreamManager, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        title = QLabel("Настройки")
        title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 24px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        layout.addWidget(title)

        hint = QLabel(
            "Выбор пайплайна и параметров обработки — во вкладке «Пресеты». "
            "При добавлении потока выберите подходящий пресет."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; background: transparent; border: none;"
        )
        layout.addWidget(hint)

        # Справка по доступным пайплайнам.
        info_title = QLabel("Доступные пайплайны")
        info_title.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 14px; font-weight: 600;"
            " background: transparent; border: none;"
        )
        layout.addWidget(info_title)

        for info in get_pipelines():
            name = QLabel(info.label)
            name.setStyleSheet(
                f"color: {T.TEXT_PRIMARY}; font-weight: 600; background: transparent;"
                " border: none;"
            )
            desc = QLabel(info.description)
            desc.setWordWrap(True)
            desc.setStyleSheet(
                f"color: {T.TEXT_MUTED}; font-size: 12px; background: transparent;"
                " border: none;"
            )
            layout.addWidget(name)
            layout.addWidget(desc)

        layout.addStretch(1)
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
