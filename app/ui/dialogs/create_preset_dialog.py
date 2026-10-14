"""Диалог создания/редактирования пресета обработки.

Позволяет задать имя, выбрать пайплайн и настроить его параметры. Поля
параметров генерируются автоматически на основе значений по умолчанию
пайплайна: int -> QSpinBox, float -> QDoubleSpinBox, str -> QLineEdit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.core.pipeline_registry import get_pipeline, get_pipelines
from app.core.preset_manager import Preset


class CreatePresetDialog(QDialog):
    """Модальное окно создания/редактирования пресета."""

    def __init__(self, parent=None, preset: Optional[Preset] = None) -> None:
        super().__init__(parent)
        self._preset = preset
        self.setWindowTitle("Новый пресет" if preset is None else "Изменить пресет")
        self.setMinimumWidth(480)
        self.setStyleSheet(f"background: {T.BG_BASE};")

        self._result: Optional[Tuple[str, str, Dict[str, Any]]] = None
        self._param_widgets: Dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel(self.windowTitle())
        title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 18px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        layout.addWidget(title)

        # --- Имя ---
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(12)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Например: Быстрый анализ")
        if preset is not None:
            self.name_edit.setText(preset.name)
        form.addRow(self._label("Название"), self.name_edit)

        # --- Пайплайн ---
        self.pipeline_combo = QComboBox()
        for info in get_pipelines():
            self.pipeline_combo.addItem(info.label, info.key)
        if preset is not None:
            idx = self.pipeline_combo.findData(preset.pipeline_key)
            if idx >= 0:
                self.pipeline_combo.setCurrentIndex(idx)
        form.addRow(self._label("Пайплайн"), self.pipeline_combo)
        layout.addLayout(form)

        self.pipeline_combo.currentIndexChanged.connect(self._rebuild_params)

        # --- Параметры (прокручиваемые) ---
        params_title = QLabel("Параметры")
        params_title.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 13px; font-weight: 600;"
            " background: transparent; border: none;"
        )
        layout.addWidget(params_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")
        self._params_container = QWidget()
        self._params_layout = QFormLayout(self._params_container)
        self._params_layout.setLabelAlignment(Qt.AlignRight)
        self._params_layout.setSpacing(10)
        scroll.setWidget(self._params_container)
        layout.addWidget(scroll, 1)

        # --- Кнопки ---
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Save).setObjectName("PrimaryButton")
        layout.addWidget(buttons)

        self._rebuild_params()

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; background: transparent; border: none;"
        )
        return lbl

    def _rebuild_params(self) -> None:
        """Перестраивает поля параметров под выбранный пайплайн."""
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._param_widgets.clear()

        key = self.pipeline_combo.currentData()
        info = get_pipeline(key) if key else None
        defaults: Dict[str, Any] = dict(info.defaults) if info else {}
        current_params: Dict[str, Any] = (
            dict(self._preset.params)
            if self._preset is not None and self._preset.pipeline_key == key
            else {}
        )
        for pname, default in defaults.items():
            widget = self._make_param_widget(current_params.get(pname, default))
            self._param_widgets[pname] = widget
            self._params_layout.addRow(self._label(pname), widget)

    @staticmethod
    def _make_param_widget(value: Any) -> QWidget:
        """Создаёт виджет ввода под тип значения параметра."""
        if isinstance(value, bool):
            combo = QComboBox()
            combo.addItem("Да", True)
            combo.addItem("Нет", False)
            combo.setCurrentIndex(0 if value else 1)
            return combo
        if isinstance(value, int):
            spin = QSpinBox()
            spin.setRange(-1_000_000, 10_000_000)
            spin.setValue(int(value))
            return spin
        if isinstance(value, float):
            spin = QDoubleSpinBox()
            spin.setRange(-1_000_000.0, 10_000_000.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.01)
            spin.setValue(float(value))
            return spin
        return QLineEdit(str(value))

    def _collect_params(self) -> Dict[str, Any]:
        """Собирает значения параметров из виджетов."""
        result: Dict[str, Any] = {}
        for pname, widget in self._param_widgets.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                result[pname] = widget.value()
            elif isinstance(widget, QComboBox):
                result[pname] = widget.currentData()
            elif isinstance(widget, QLineEdit):
                result[pname] = widget.text()
        return result

    def _on_accept(self) -> None:
        """Валидирует ввод и сохраняет результат."""
        name = self.name_edit.text().strip()
        if not name:
            self.name_edit.setFocus()
            self.name_edit.setStyleSheet(
                f"border: 1px solid {T.STATUS_DANGER}; border-radius: 8px;"
                " padding: 6px 10px;"
            )
            return
        pipeline_key = self.pipeline_combo.currentData() or "yolo_vit"
        self._result = (name, pipeline_key, self._collect_params())
        self.accept()

    def result(self) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Возвращает (name, pipeline_key, params) или None."""
        return self._result

