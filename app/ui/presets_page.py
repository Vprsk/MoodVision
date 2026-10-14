"""Вкладка «Пресеты»: управление пресетами обработки видеопотоков.

Здесь можно выбрать пайплайн, настроить его параметры, создать кастомный
пресет и сохранить его. Сохранённые пресеты затем выбираются при добавлении
видеопотока. Встроенные пресеты (по одному на пайплайн) доступны только для
просмотра.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.core.preset_manager import Preset, PresetManager
from app.ui.icons import make_plus_icon


class PresetsPage(QWidget):
    """Страница управления пресетами."""

    def __init__(self, preset_manager: PresetManager, parent=None) -> None:
        super().__init__(parent)
        self.presets = preset_manager
        self._selected_id: Optional[str] = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Левая колонка: список пресетов ---
        left = QWidget()
        left.setFixedWidth(320)
        left.setStyleSheet(f"background: {T.BG_DARKEST};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(16, 16, 16, 16)
        ll.setSpacing(12)

        top = QHBoxLayout()
        title = QLabel("Пресеты")
        title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 20px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        top.addWidget(title)
        top.addStretch(1)
        add_btn = QPushButton()
        add_btn.setObjectName("PrimaryButton")
        add_btn.setFixedSize(34, 34)
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.setToolTip("Создать пресет")
        add_btn.setIcon(make_plus_icon(box=22, color="#ffffff"))
        add_btn.setIconSize(QSize(20, 20))
        add_btn.clicked.connect(self._on_create)
        top.addWidget(add_btn)
        ll.addLayout(top)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            f"""
            QListWidget {{ background: transparent; border: none; }}
            QListWidget::item {{ padding: 8px; border-radius: 8px; }}
            QListWidget::item:selected {{ background: {T.BG_CARD_HOVER};
                color: {T.TEXT_PRIMARY}; }}
            """
        )
        self.list_widget.currentItemChanged.connect(self._on_selected)
        ll.addWidget(self.list_widget, 1)
        root.addWidget(left)

        # --- Правая колонка: детали пресета ---
        self._detail = self._build_detail()
        root.addWidget(self._detail, 1)

        self.refresh()

    def _build_detail(self) -> QWidget:
        """Строит правую панель с деталями выбранного пресета."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        head = QHBoxLayout()
        self._detail_title = QLabel("Выберите пресет")
        self._detail_title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 22px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        head.addWidget(self._detail_title)
        head.addStretch(1)
        self._edit_btn = QPushButton("Изменить")
        self._edit_btn.clicked.connect(self._on_edit)
        self._edit_btn.setEnabled(False)
        self._delete_btn = QPushButton("Удалить")
        self._delete_btn.setObjectName("DangerButton")
        self._delete_btn.clicked.connect(self._on_delete)
        self._delete_btn.setEnabled(False)
        head.addWidget(self._edit_btn)
        head.addWidget(self._delete_btn)
        layout.addLayout(head)

        self._pipeline_label = QLabel("")
        self._pipeline_label.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 13px;"
            " background: transparent; border: none;"
        )
        layout.addWidget(self._pipeline_label)

        self._desc_label = QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: 12px; background: transparent;"
            " border: none;"
        )
        layout.addWidget(self._desc_label)

        # Таблица параметров (прокручиваемая).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")
        self._params_container = QWidget()
        self._params_container.setStyleSheet("background: transparent;")
        self._params_layout = QVBoxLayout(self._params_container)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        self._params_layout.setSpacing(6)
        self._params_layout.addStretch(1)
        scroll.setWidget(self._params_container)
        layout.addWidget(scroll, 1)

        return panel

    # --- Обновление списка и деталей ---------------------------------------

    def refresh(self) -> None:
        """Перерисовывает список пресетов (встроенные + кастомные)."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for preset in self.presets.all():
            item = QListWidgetItem(preset.name)
            item.setData(Qt.UserRole, preset.id)
            if not preset.is_custom:
                item.setToolTip("Встроенный пресет (только просмотр)")
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

        # Восстановим выбор, если он ещё существует, иначе выберем первый.
        if self._selected_id and self.presets.get(self._selected_id):
            self._select_by_id(self._selected_id)
        elif self.list_widget.count():
            self.list_widget.setCurrentRow(0)
        else:
            self._selected_id = None
            self._render_detail(None)

    def _select_by_id(self, preset_id: str) -> None:
        """Выделяет элемент списка по id пресета."""
        for row in range(self.list_widget.count()):
            if self.list_widget.item(row).data(Qt.UserRole) == preset_id:
                self.list_widget.setCurrentRow(row)
                return

    def _on_selected(self, current, _previous) -> None:
        if current is None:
            self._selected_id = None
            self._render_detail(None)
            return
        self._selected_id = current.data(Qt.UserRole)
        self._render_detail(self.presets.get(self._selected_id))

    def _render_detail(self, preset: Optional[Preset]) -> None:
        """Отрисовывает детали выбранного пресета."""
        # Очищаем таблицу параметров.
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if preset is None:
            self._detail_title.setText("Выберите пресет")
            self._pipeline_label.setText("")
            self._desc_label.setText("")
            self._edit_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            self._params_layout.addStretch(1)
            return

        self._detail_title.setText(preset.name)
        self._pipeline_label.setText(f"Пайплайн: {preset.pipeline_key}")
        if preset.description:
            self._desc_label.setText(preset.description)
        elif preset.is_custom:
            self._desc_label.setText("Кастомный пресет.")
        else:
            self._desc_label.setText("")

        is_custom = preset.is_custom
        self._edit_btn.setEnabled(is_custom)
        self._delete_btn.setEnabled(is_custom)

        if not preset.params:
            empty = QLabel("Нет настраиваемых параметров.")
            empty.setStyleSheet(
                f"color: {T.TEXT_MUTED}; background: transparent; border: none;"
            )
            self._params_layout.addWidget(empty)
        else:
            for pname, value in preset.params.items():
                row = self._make_param_row(pname, value)
                self._params_layout.addWidget(row)
        self._params_layout.addStretch(1)

    @staticmethod
    def _make_param_row(name: str, value) -> QFrame:
        """Создаёт строку «параметр = значение» для панели деталей."""
        row = QFrame()
        row.setObjectName("Card")
        row.setStyleSheet(
            f"background: {T.BG_CARD}; border: 1px solid {T.BORDER};"
            " border-radius: 8px;"
        )
        lay = QHBoxLayout(row)
        lay.setContentsMargins(12, 8, 12, 8)
        name_lbl = QLabel(str(name))
        name_lbl.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; background: transparent; border: none;"
        )
        value_lbl = QLabel(str(value))
        value_lbl.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-weight: 600; background: transparent;"
            " border: none;"
        )
        lay.addWidget(name_lbl, 1)
        lay.addWidget(value_lbl)
        return row

    # --- Создание / редактирование / удаление ------------------------------

    def _on_create(self) -> None:
        """Открывает диалог создания пресета и сохраняет его."""
        from app.ui.dialogs.create_preset_dialog import CreatePresetDialog

        dialog = CreatePresetDialog(self)
        if dialog.exec() and dialog.result() is not None:
            name, pipeline_key, params = dialog.result()
            preset = self.presets.create(name, pipeline_key, params)
            self._selected_id = preset.id
            self._save_and_refresh()

    def _on_edit(self) -> None:
        """Открывает диалог редактирования выбранного кастомного пресета."""
        preset = self.presets.get(self._selected_id) if self._selected_id else None
        if preset is None or not preset.is_custom:
            return
        from app.ui.dialogs.create_preset_dialog import CreatePresetDialog

        dialog = CreatePresetDialog(self, preset=preset)
        if dialog.exec() and dialog.result() is not None:
            name, pipeline_key, params = dialog.result()
            self.presets.update(
                preset.id, name=name, pipeline_key=pipeline_key, params=params
            )
            self._save_and_refresh()

    def _on_delete(self) -> None:
        """Удаляет выбранный кастомный пресет (с подтверждением)."""
        preset = self.presets.get(self._selected_id) if self._selected_id else None
        if preset is None or not preset.is_custom:
            return
        if (
            QMessageBox.question(
                self, "Удалить пресет", f"Удалить пресет «{preset.name}»?"
            )
            != QMessageBox.Yes
        ):
            return
        self.presets.delete(preset.id)
        self._selected_id = None
        self._save_and_refresh()

    def _save_and_refresh(self) -> None:
        """Сохраняет пресеты на диск и обновляет интерфейс."""
        try:
            self.presets.save()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Пресеты", f"Не удалось сохранить: {exc}")
        self.refresh()


