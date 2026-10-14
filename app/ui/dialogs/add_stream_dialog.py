"""Диалог добавления нового видеопотока.

Позволяет задать имя, источник (RTSP/HTTP-URL, путь к файлу, веб-камера),
локацию и выбрать пайплайн обработки из реестра. Возвращает готовый
StreamConfig через result_config().
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.core.group_manager import DEFAULT_GROUP
from app.core.pipeline_registry import get_pipelines
from app.core.stream_config import StreamConfig


class AddStreamDialog(QDialog):
    """Модальное окно добавления потока."""

    def __init__(
        self,
        parent=None,
        groups: Optional[List[str]] = None,
        default_group: Optional[str] = None,
        preset_manager=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Добавить поток")
        self.setMinimumWidth(460)
        self.setStyleSheet(f"background: {T.BG_BASE};")

        self._config: Optional[StreamConfig] = None
        self._preset_manager = preset_manager

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("Новый видеопоток")
        title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 18px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        layout.addWidget(title)

        hint = QLabel(
            "Источник: RTSP/HTTP-ссылка, путь к видеофайлу или "
            "«webcam:0» для веб-камеры."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 12px; background: transparent;"
            " border: none;"
        )
        layout.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignRight)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Например: Ресепшн")
        self.name_edit.setText("Новый поток")
        form.addRow(self._label("Название"), self.name_edit)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("rtsp://...  |  C:/video.mp4  |  webcam:0")
        form.addRow(self._label("Источник"), self.source_edit)

        # Группа: выбор из существующих или ввод новой (editable combo).
        self.group_combo = QComboBox()
        self.group_combo.setEditable(True)
        self.group_combo.setInsertPolicy(QComboBox.NoInsert)
        for gname in (groups or [DEFAULT_GROUP]):
            self.group_combo.addItem(gname)
        self.group_combo.setPlaceholderText("Например: Тверская")
        if default_group:
            idx = self.group_combo.findText(default_group)
            if idx >= 0:
                self.group_combo.setCurrentIndex(idx)
            else:
                self.group_combo.setEditText(default_group)
        form.addRow(self._label("Группа"), self.group_combo)

        # Пресет: именованный набор «пайплайн + параметры».
        self.preset_combo = QComboBox()
        if self._preset_manager is not None:
            for preset in self._preset_manager.all():
                self.preset_combo.addItem(preset.name, preset.id)
        else:
            # Фоллбэк: без менеджера — встроенные пайплайны как пресеты.
            for info in get_pipelines():
                self.preset_combo.addItem(info.label, f"builtin:{info.key}")
        form.addRow(self._label("Пресет"), self.preset_combo)

        layout.addLayout(form)

        # --- Кнопки ---
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Отмена")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Добавить")
        ok.setObjectName("PrimaryButton")
        ok.clicked.connect(self._on_accept)
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        layout.addLayout(buttons)

        self.source_edit.setFocus()

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; background: transparent; border: none;"
        )
        return lbl

    def _on_accept(self) -> None:
        """Валидирует ввод и формирует StreamConfig на основе пресета."""
        source = self.source_edit.text().strip()
        if not source:
            self.source_edit.setFocus()
            self.source_edit.setStyleSheet(
                f"border: 1px solid {T.STATUS_DANGER}; border-radius: 8px;"
                " padding: 6px 10px;"
            )
            return
        # Резолвим выбранный пресет в (pipeline_key, params).
        preset_id = self.preset_combo.currentData()
        if self._preset_manager is not None:
            pipeline_key, params = self._preset_manager.resolve(preset_id)
        else:
            # Фоллбэк без менеджера: параметры по умолчанию пайплайна.
            from app.core.pipeline_registry import get_pipeline

            pipeline_key = (preset_id or "builtin:yolo_vit").split("builtin:")[-1]
            info = get_pipeline(pipeline_key)
            params = dict(info.defaults) if info else {}
        self._config = StreamConfig.create(
            name=self.name_edit.text(),
            source=source,
            location=self.group_combo.currentText().strip(),
            pipeline_key=pipeline_key,
            pipeline_params=params,
        )
        self.accept()

    def result_config(self) -> Optional[StreamConfig]:
        """Возвращает созданный StreamConfig (если диалог принят)."""
        return self._config
