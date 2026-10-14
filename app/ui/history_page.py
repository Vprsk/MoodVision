"""Страница «История»: записи обработанных видео.

Слева — зона загрузки (drag-and-drop) и список записей.
Справа — статистика выбранной записи. На данном этапе отображает
загруженные файлы и их метаданные; глубокая аналитика эпизодов
подключается к результатам пакетной обработки записей.
"""

from __future__ import annotations

import os
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.ui.widgets.badges import MetricCard


class _DropZone(QFrame):
    """Зона перетаскивания видеофайлов."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setAcceptDrops(True)
        self.setMinimumHeight(220)
        self._normal = (
            f"background: {T.BG_CARD}; border: 2px dashed {T.BORDER_STRONG};"
            " border-radius: 12px;"
        )
        self.setStyleSheet(self._normal)
        self.files_dropped = None  # устанавливается страницей

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)

        icon = QLabel("⬆")
        icon.setStyleSheet(
            f"font-size: 34px; color: {T.TEXT_MUTED}; background: transparent;"
            " border: none;"
        )
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        text = QLabel("Перетащите видео сюда")
        text.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; background: transparent; border: none;"
        )
        text.setAlignment(Qt.AlignCenter)
        layout.addWidget(text)

        self.choose_btn = QPushButton("Выбрать файлы")
        self.choose_btn.setObjectName("PrimaryButton")
        self.choose_btn.setFixedWidth(160)
        layout.addWidget(self.choose_btn, 0, Qt.AlignCenter)

        fmt = QLabel("MP4, AVI, MOV, MKV")
        fmt.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: 11px; background: transparent;"
            " border: none;"
        )
        fmt.setAlignment(Qt.AlignCenter)
        layout.addWidget(fmt)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                f"background: {T.BG_CARD_HOVER}; border: 2px dashed {T.ACCENT};"
                " border-radius: 12px;"
            )

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self.setStyleSheet(self._normal)

    def dropEvent(self, event) -> None:  # noqa: N802
        self.dragLeaveEvent(event)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths and self.files_dropped:
            self.files_dropped(paths)


class HistoryPage(QWidget):
    """Страница истории записей."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records: List[str] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Левая колонка: загрузка + список ---
        left = QWidget()
        left.setFixedWidth(360)
        left.setStyleSheet(f"background: {T.BG_DARKEST};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(16, 16, 16, 16)
        ll.setSpacing(12)

        title = QLabel("Записи")
        title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 20px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        ll.addWidget(title)

        self.drop = _DropZone()
        self.drop.choose_btn.clicked.connect(self._on_choose)
        ll.addWidget(self.drop)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            f"""
            QListWidget {{ background: transparent; border: none; }}
            QListWidget::item {{ padding: 6px; }}
            QListWidget::item:selected {{ background: {T.BG_CARD_HOVER};
                border-radius: 8px; }}
            """
        )
        self.list_widget.currentTextChanged.connect(self._on_selected)
        ll.addWidget(self.list_widget, 1)
        root.addWidget(left)

        # --- Правая колонка: статистика ---
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(24, 20, 24, 20)
        rl.setSpacing(16)

        self.detail_title = QLabel("Выберите запись")
        self.detail_title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 22px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        rl.addWidget(self.detail_title)

        cards = QHBoxLayout()
        cards.setSpacing(14)
        self.card_incidents = MetricCard("Инцидентов", "—", "эпизодов")
        self.card_unsure = MetricCard("«Не уверена»", "—", "эпизодов")
        self.card_useful = MetricCard("Пригодных кадров", "—", "по всей записи")
        cards.addWidget(self.card_incidents)
        cards.addWidget(self.card_unsure)
        cards.addWidget(self.card_useful)
        cards.addStretch(1)
        rl.addLayout(cards)

        placeholder = QLabel(
            "Загрузите видеозапись, чтобы запустить пакетный анализ эмоций. "
            "Здесь появятся эпизоды, таймлайн и сводка по клиентам."
        )
        placeholder.setWordWrap(True)
        placeholder.setStyleSheet(
            f"color: {T.TEXT_MUTED}; background: transparent; border: none;"
        )
        placeholder.setAlignment(Qt.AlignCenter)
        rl.addWidget(placeholder, 1)

        root.addWidget(right, 1)

        # Связываем зону загрузки с обработчиком страницы.
        self.drop.files_dropped = self._on_files_dropped

    def _on_choose(self) -> None:
        """Открывает диалог выбора видеофайлов."""
        from PySide6.QtWidgets import QFileDialog

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите видео",
            "",
            "Видео (*.mp4 *.avi *.mov *.mkv);;Все файлы (*)",
        )
        if paths:
            self._on_files_dropped(paths)

    def _on_files_dropped(self, paths: List[str]) -> None:
        """Добавляет файлы в список записей."""
        for path in paths:
            if path not in self._records:
                self._records.append(path)
                QListWidgetItem(os.path.basename(path), self.list_widget)

    def _on_selected(self, name: str) -> None:
        """Обновляет заголовок детальной панели по выбранной записи."""
        if name:
            self.detail_title.setText(name)

