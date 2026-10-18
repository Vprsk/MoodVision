"""Страница «Настройки»: параметры камер, пресеты, справка."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.core.logging import get_logger
from app.core.persistence import save_streams
from app.core.stream_config import StreamConfig
from app.core.stream_manager import StreamManager
from app.ui.widgets.segmented import SegmentedControl, ToggleSwitch

logger = get_logger("settings_page")

SECTIONS: List[tuple] = [
    ("notifications", "Уведомления"),
    ("recipients", "Получатели"),
    ("cameras", "Камеры"),
    ("journal", "Журнал и данные"),
    ("presets", "Пресеты"),
]

_CAMERA_ROWS = [
    ("Поток для поиска лиц", "face_stream", ["Субпоток", "Основной"]),
    ("Размер кадра для поиска", "search_frame",
     ["Авто", "384", "640", "960", "1280", "Тайлы 2×2"]),
    ("Кроп лица", "face_crop", ["160×160", "224×224", "260×260"]),
    ("Частота анализа", "analysis_fps", ["3 к/с", "5 к/с", "8 к/с"]),
    ("Мин. размер лица", "min_face", ["40 px", "48 px", "56 px", "64 px"]),
    ("Мин. резкость", "sharpness", ["Низкая", "Средняя", "Высокая"]),
    ("Макс. поворот головы", "head_yaw", ["30°", "45°", "60°"]),
    ("Поворот кадра", "frame_rotation", ["0°", "90°", "180°", "270°"]),
]

_UI_DEFAULTS = {
    "face_stream": "Субпоток",
    "search_frame": "Авто",
    "face_crop": "224×224",
    "analysis_fps": "5 к/с",
    "min_face": "56 px",
    "sharpness": "Средняя",
    "head_yaw": "45°",
    "frame_rotation": "0°",
}


class SettingsPage(QWidget):
    """Настройки приложения."""

    def __init__(
        self,
        manager: StreamManager,
        preset_manager=None,
        presets_page_factory: Optional[Callable[[], QWidget]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.manager = manager
        self._presets_page_factory = presets_page_factory

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        left = QWidget()
        left.setFixedWidth(260)
        left.setStyleSheet(f"background: {T.BG_DARKEST};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(16, 24, 16, 16)
        ll.setSpacing(4)

        self._section_buttons: Dict[str, QPushButton] = {}
        for key, label in SECTIONS:
            btn = QPushButton(label)
            btn.setObjectName("SettingsNavButton")
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.clicked.connect(lambda _=False, k=key: self._show_section(k))
            self._section_buttons[key] = btn
            ll.addWidget(btn)
        ll.addStretch(1)
        root.addWidget(left)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_info_section(
            "Уведомления",
            "Настройка оповещений о инцидентах: звуки, всплывающие окна, "
            "порог уверенности для алертов.",
        ))
        self._stack.addWidget(self._build_info_section(
            "Получатели",
            "Список получателей уведомлений: email, Telegram-бот, webhook. "
            "Уведомления отправляются при подтверждённых инцидентах.",
        ))
        self._stack.addWidget(self._build_cameras_section())
        self._stack.addWidget(self._build_info_section(
            "Журнал и данные",
            "Хранение результатов анализа, экспорт журналов в CSV/JSON, "
            "очистка кэша и логов приложения.",
        ))
        if presets_page_factory is not None:
            self._stack.addWidget(presets_page_factory())
        else:
            self._stack.addWidget(self._build_info_section(
                "Пресеты", "Пресеты недоступны."
            ))
        root.addWidget(self._stack, 1)

        # Начальная секция — «Уведомления» (индекс 0 стека).
        self._show_section("notifications")

    def _show_section(self, key: str) -> None:
        order = [k for k, _ in SECTIONS]
        if key not in order:
            return
        for k, btn in self._section_buttons.items():
            btn.setChecked(k == key)
        self._stack.setCurrentIndex(order.index(key))
        if key == "cameras":
            self._rebuild_camera_list()

    @staticmethod
    def _build_info_section(title: str, text: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(12)

        heading = QLabel(title)
        heading.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 22px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        layout.addWidget(heading)

        desc = QLabel(text)
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; background: transparent; border: none;"
        )
        layout.addWidget(desc)
        layout.addStretch(1)
        return page

    def _build_cameras_section(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(36, 24, 36, 24)
        outer.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel("Камеры")
        title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 22px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        header.addWidget(title)
        header.addStretch(1)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Поиск")
        self._search.setClearButtonEnabled(True)
        self._search.setMaximumWidth(320)
        self._search.textChanged.connect(lambda _t: self._rebuild_camera_list())
        header.addWidget(self._search)
        outer.addLayout(header)

        top_row = QHBoxLayout()
        self._select_all = QPushButton("☐  Все камеры")
        self._select_all.setObjectName("FlatButton")
        self._select_all.setCursor(QCursor(Qt.PointingHandCursor))
        self._select_all.clicked.connect(self._toggle_select_all)
        top_row.addWidget(self._select_all)
        self._all_count = QLabel("0")
        self._all_count.setStyleSheet(
            f"color: {T.TEXT_MUTED}; background: transparent; border: none;"
        )
        top_row.addWidget(self._all_count)
        top_row.addStretch(1)
        tune_btn = QPushButton("Настроить выбранные")
        tune_btn.setObjectName("FlatButton")
        top_row.addWidget(tune_btn)
        outer.addLayout(top_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")
        self._list_container = QWidget()
        self._list_container.setStyleSheet("background: transparent;")
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(8)
        self._list_layout.addStretch(1)
        scroll.setWidget(self._list_container)
        outer.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Отмена")
        cancel.clicked.connect(self._rebuild_camera_list)
        save = QPushButton("Сохранить")
        save.setObjectName("PrimaryButton")
        save.clicked.connect(self._save_camera_settings)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        outer.addLayout(buttons)

        self._camera_rows: List = []
        return page

    def _rebuild_camera_list(self) -> None:
        if not hasattr(self, "_list_layout"):
            return
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._camera_rows = []

        query = (self._search.text() if hasattr(self, "_search") else "").strip().lower()
        visible = 0
        for runtime in self.manager.all_runtimes():
            config = runtime.config
            if query and query not in config.name.lower() and \
                    query not in (config.location or "").lower():
                continue
            row = _CameraSettingRow(config, runtime.is_live)
            row.expanded_changed.connect(
                lambda exp, r=row: self._collapse_others(r)
            )
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)
            self._camera_rows.append(row)
            visible += 1

        self._list_layout.addStretch(1)
        if self._all_count is not None:
            self._all_count.setText(str(visible))

    def _collapse_others(self, keep: "_CameraSettingRow") -> None:
        for row in self._camera_rows:
            if row is not keep:
                row.set_expanded(False)

    def _toggle_select_all(self) -> None:
        any_unchecked = any(not row.checked() for row in self._camera_rows)
        for row in self._camera_rows:
            row.set_checked(any_unchecked)
        self._select_all.setText(
            "☑  Все камеры" if any_unchecked else "☐  Все камеры"
        )

    def _save_camera_settings(self) -> None:
        for row in self._camera_rows:
            row.apply()
        configs: List[StreamConfig] = [
            rt.config for rt in self.manager.all_runtimes()
        ]
        try:
            save_streams(configs)
            logger.info("Настройки камер сохранены (%d)", len(configs))
        except Exception as exc:  # noqa: BLE001
            logger.error("Не удалось сохранить настройки камер: %s", exc)
        self._rebuild_camera_list()


class _CameraSettingRow(QWidget):
    expanded_changed = Signal(bool)

    def __init__(self, config: StreamConfig, live: bool, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._expanded = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        head = QFrame()
        head.setObjectName("Card")
        head.setStyleSheet(
            f"background: {T.BG_CARD}; border: 1px solid {T.BORDER};"
            " border-radius: 10px;"
        )
        hl = QHBoxLayout(head)
        hl.setContentsMargins(16, 12, 16, 12)
        hl.setSpacing(12)

        self._checkbox = QPushButton("☐")
        self._checkbox.setFixedSize(26, 26)
        self._checkbox.setObjectName("FlatButton")
        self._checkbox.setCursor(QCursor(Qt.PointingHandCursor))
        self._checkbox.clicked.connect(self._toggle_checked)
        hl.addWidget(self._checkbox)

        name_row = QVBoxLayout()
        name_row.setSpacing(2)
        name_lbl = QLabel(config.name)
        name_lbl.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-weight: 600; font-size: 14px;"
            " background: transparent; border: none;"
        )
        name_row.addWidget(name_lbl)
        if config.location:
            loc = QLabel(config.location)
            loc.setStyleSheet(
                f"color: {T.TEXT_MUTED}; font-size: 12px; background: transparent;"
                " border: none;"
            )
            name_row.addWidget(loc)
        hl.addLayout(name_row, 1)

        ui = dict(_UI_DEFAULTS)
        ui.update(config.ui_settings or {})
        self._summary = QLabel(
            f"{ui['search_frame']}  ·  {ui['analysis_fps']}"
            if config.enabled else "выключен"
        )
        self._summary.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 12px; background: transparent;"
            " border: none;"
        )
        hl.addWidget(self._summary)

        self._chevron = QLabel("⌄")
        self._chevron.setStyleSheet(
            f"color: {T.TEXT_MUTED}; background: transparent; border: none;"
            " font-size: 16px;"
        )
        hl.addWidget(self._chevron)

        head.mousePressEvent = self._on_head_clicked  # type: ignore[method-assign]
        root.addWidget(head)

        self._panel = QWidget()
        self._panel.setObjectName("Card")
        self._panel.setStyleSheet(
            f"background: {T.BG_BASE}; border: 1px solid {T.BORDER};"
            " border-radius: 10px;"
        )
        pl = QVBoxLayout(self._panel)
        pl.setContentsMargins(20, 14, 20, 14)
        pl.setSpacing(2)

        analyze_row = QHBoxLayout()
        analyze_row.setContentsMargins(0, 8, 0, 8)
        analyze_lbl = QLabel("Анализ")
        analyze_lbl.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; background: transparent; border: none;"
        )
        analyze_row.addWidget(analyze_lbl)
        analyze_row.addStretch(1)
        self._analyze_toggle = ToggleSwitch(config.enabled)
        self._analyze_toggle.valueChanged.connect(self._on_toggled_analyze)
        analyze_row.addWidget(self._analyze_toggle)
        pl.addLayout(analyze_row)

        self._segments: Dict[str, SegmentedControl] = {}
        for label, key, options in _CAMERA_ROWS:
            row = QHBoxLayout()
            row.setContentsMargins(0, 8, 0, 8)
            lbl = QLabel(label)
            lbl.setStyleSheet(
                f"color: {T.TEXT_SECONDARY}; background: transparent; border: none;"
            )
            row.addWidget(lbl)
            row.addStretch(1)
            seg = SegmentedControl(options, ui.get(key, _UI_DEFAULTS[key]))
            self._segments[key] = seg
            row.addWidget(seg)
            pl.addLayout(row)

        addr_row = QHBoxLayout()
        addr_row.setContentsMargins(0, 8, 0, 8)
        addr_lbl = QLabel("Адрес потока")
        addr_lbl.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; background: transparent; border: none;"
        )
        addr_row.addWidget(addr_lbl)
        addr_row.addStretch(1)
        self._source_edit = QLineEdit(config.source)
        self._source_edit.setMaximumWidth(360)
        addr_row.addWidget(self._source_edit)
        pl.addLayout(addr_row)

        self._panel.setVisible(False)
        root.addWidget(self._panel)

    def _on_head_clicked(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.set_expanded(not self._expanded)
            self.expanded_changed.emit(self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._panel.setVisible(expanded)
        self._chevron.setText("⌃" if expanded else "⌄")

    def checked(self) -> bool:
        return self._checkbox.text() == "☑"

    def set_checked(self, checked: bool) -> None:
        self._checkbox.setText("☑" if checked else "☐")

    def _toggle_checked(self) -> None:
        self.set_checked(not self.checked())

    def _on_toggled_analyze(self, checked: bool) -> None:
        self._config.enabled = checked
        self._summary.setText(
            f"{self._segments['search_frame'].value()}  ·  "
            f"{self._segments['analysis_fps'].value()}"
            if checked else "выключен"
        )

    def apply(self) -> None:
        if not self.checked():
            return
        self._config.enabled = self._analyze_toggle.isChecked()
        self._config.source = self._source_edit.text().strip()
        settings = dict(self._config.ui_settings or {})
        for key, seg in self._segments.items():
            settings[key] = seg.value()
        self._config.ui_settings = settings
        self._show_section("cameras")
