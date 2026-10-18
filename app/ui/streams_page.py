"""Страница «Потоки»: живой мониторинг всех видеопотоков.

Слева — список камер по локациям (выбор/переключение/добавление).
Справа — крупный LIVE-поток выбранной камеры с панелью посетителей
и строкой метрик. Страница напрямую подключается к сигналу frame_ready
каждого VideoWorker, чтобы показывать кадры с минимальной задержкой.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.core.group_manager import DEFAULT_GROUP, GroupManager
from app.core.logging import get_logger
from app.core.stream_config import (
    STATUS_INCIDENT,
    STATUS_NORMAL,
    STATUS_UNSURE,
    ClientTrack,
    StreamConfig,
)
from app.core.stream_manager import StreamManager, StreamRuntime
from app.ui.icons import make_plus_icon
from app.ui.widgets.badges import Badge, MetricCard
from app.ui.widgets.incident_banner import IncidentBanner
from app.ui.widgets.video_player import VideoPlayer
from app.ui.widgets.visitors_panel import VisitorsPanel

logger = get_logger("streams_page")

# Как часто (в секундах) перерисовывать тяжёлую панель «Посетители».
# Сигнал треков приходит на каждом кадре, но панель обновляем реже.
VISITORS_REFRESH_SECONDS = 0.2  # ~5 раз в секунду

# Подпись под метрикой FPS (заменяет «Пригодных кадров», когда нет видео).
FPS_CAPTION = "FPS обработки"


class _CameraItem(QFrame):
    """Элемент списка камер: имя, цвет-статус, текстовый статус, бейдж."""

    menu_requested = Signal()  # нажали «⋯» — показать меню действий с потоком

    def __init__(self, runtime: StreamRuntime, parent=None) -> None:
        super().__init__(parent)
        self.stream_id = runtime.config.stream_id
        self.setObjectName("Card")
        self.setStyleSheet(
            f"background: {T.BG_CARD}; border: 1px solid {T.BORDER};"
            " border-radius: 10px;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        self._dot = QLabel("●")
        self._dot.setStyleSheet(f"color: {T.STATUS_IDLE}; font-size: 14px;")
        layout.addWidget(self._dot)

        col = QVBoxLayout()
        col.setSpacing(2)
        self._name = QLabel(runtime.config.name)
        self._name.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-weight: 600; background: transparent;"
            " border: none;"
        )
        self._status = QLabel("Выключено")
        self._status.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: 12px; background: transparent;"
            " border: none;"
        )
        col.addWidget(self._name)
        col.addWidget(self._status)
        layout.addLayout(col, 1)

        self._badge = Badge("—", T.TEXT_MUTED)
        layout.addWidget(self._badge)

        # Кнопка «⋯» — заметный способ открыть меню действий (переместить,
        # переименовать, удалить), помимо правого клика по карточке.
        self._menu_btn = QPushButton("⋯")
        self._menu_btn.setFixedSize(26, 26)
        self._menu_btn.setCursor(Qt.PointingHandCursor)
        self._menu_btn.setToolTip("Действия с потоком")
        self._menu_btn.setStyleSheet(
            f"background: transparent; border: none; color: {T.TEXT_MUTED};"
            " font-size: 16px; border-radius: 6px;"
        )
        self._menu_btn.clicked.connect(self.menu_requested.emit)
        layout.addWidget(self._menu_btn)

        self.setCursor(Qt.PointingHandCursor)

    def menu_anchor(self) -> QPoint:
        """Глобальная точка, откуда открыть меню (под кнопкой «⋯»)."""
        return self._menu_btn.mapToGlobal(self._menu_btn.rect().bottomLeft())

    def set_status(self, status: str) -> None:
        """Обновляет индикатор статуса камеры."""
        color = T.status_color(status)
        self._dot.setStyleSheet(f"color: {color}; font-size: 14px;")
        self._status.setText(status)
        self._status.setStyleSheet(
            f"color: {color}; font-size: 12px; background: transparent; border: none;"
        )

    def set_badge(self, text: str, color: str) -> None:
        self._badge.set_badge(text, color)


class _GroupHeader(QWidget):
    """Заголовок группы в сайдбаре: стрелка сворачивания, название, «+»."""

    def __init__(self, name: str, collapsed: bool, count: int, parent=None) -> None:
        super().__init__(parent)
        self.group_name = name
        self._collapsed = collapsed
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(8)

        self._arrow = QLabel("▾" if not collapsed else "▸")
        self._arrow.setFixedWidth(14)
        self._arrow.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 12px;"
            " background: transparent; border: none;"
        )
        layout.addWidget(self._arrow)

        self._label = QLabel(name.upper())
        self._label.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: 11px; font-weight: 700;"
            " letter-spacing: 1px; background: transparent; border: none;"
        )
        layout.addWidget(self._label, 1)

        self._count = QLabel(str(count))
        self._count.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: 11px;"
            " background: transparent; border: none;"
        )
        layout.addWidget(self._count)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self._arrow.setText("▾" if not collapsed else "▸")

    def set_count(self, count: int) -> None:
        self._count.setText(str(count))



class StreamsPage(QWidget):
    """Страница живого мониторинга потоков."""

    def __init__(self, manager: StreamManager, group_manager: GroupManager, preset_manager=None, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.group_manager = group_manager
        self.preset_manager = preset_manager
        self._selected_id: Optional[str] = None
        self._camera_widgets: Dict[str, _CameraItem] = {}
        self._frame_conns: Dict[str, object] = {}
        self._selected_tracks: List[ClientTrack] = []
        # Троттлинг панели посетителей: кэш последних треков + время.
        self._pending_tracks: List[ClientTrack] = []
        self._last_visitors_update: float = 0.0
        # Для контекстного меню: какой поток/группу показали по правому клику.
        self._context_stream_id: Optional[str] = None
        self._context_group: Optional[str] = None
        self._current_pipeline_label: str = "—"

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_detail())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1120])
        root.addWidget(splitter)

        # --- Сигналы менеджера ---
        self.manager.stream_status_changed.connect(self._on_status_changed)
        self.manager.tracks_updated.connect(self._on_tracks_updated)

        self.refresh_stream_list()

    # --- Построение интерфейса ---------------------------------------------

    def _build_sidebar(self) -> QWidget:
        """Левая колонка: заголовок, кнопка добавления, список камер."""
        panel = QWidget()
        panel.setFixedWidth(300)
        panel.setStyleSheet(f"background: {T.BG_DARKEST};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(12)

        top = QHBoxLayout()
        title = QLabel("Потоки")
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
        add_btn.setToolTip("Добавить поток")
        # Красивая векторная иконка «плюс» вместо юникод-символа.
        add_btn.setIcon(make_plus_icon(box=22, color="#ffffff"))
        add_btn.setIconSize(QSize(20, 20))
        add_btn.clicked.connect(self._on_add_clicked)
        top.addWidget(add_btn)
        layout.addLayout(top)

        # Прокручиваемый список камер.
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
        layout.addWidget(scroll, 1)

        return panel

    def _build_detail(self) -> QWidget:
        """Правая колонка: баннер, метрики, видео, посетители, журнал."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # --- Баннер инцидента (сверху, по референсу) ---
        self._incident_banner = IncidentBanner()
        self._incident_banner.confirm.connect(self._on_incident_confirm)
        self._incident_banner.mark_false.connect(self._on_incident_false)
        self._incident_banner.dismiss.connect(self._on_incident_dismiss)
        layout.addWidget(self._incident_banner)

        # --- Заголовок выбранной камеры ---
        head = QHBoxLayout()
        head.setSpacing(12)
        self._detail_title = QLabel("Выберите поток")
        self._detail_title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 24px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        head.addWidget(self._detail_title)
        self._live_badge = Badge("LIVE", T.STATUS_LIVE)
        head.addWidget(self._live_badge)
        head.addStretch(1)
        self._toggle_btn = QPushButton("Включить")
        self._toggle_btn.setObjectName("PrimaryButton")
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        self._toggle_btn.setEnabled(False)
        head.addWidget(self._toggle_btn)
        layout.addLayout(head)

        # --- Карточки метрик (по референсу) ---
        cards = QHBoxLayout()
        cards.setSpacing(12)
        self._clients_metric = MetricCard("Клиентов", "0", "в кадре")
        self._status_metric = MetricCard("Статус", "—", "по всем клиентам")
        self._suitable_metric = MetricCard("Пригодных кадров", "—", "за сессию")
        cards.addWidget(self._clients_metric)
        cards.addWidget(self._status_metric)
        cards.addWidget(self._suitable_metric)
        cards.addStretch(1)
        layout.addLayout(cards)

        # --- Видео + правая панель посетителей ---
        content = QHBoxLayout()
        content.setSpacing(14)

        video_col = QVBoxLayout()
        video_col.setSpacing(8)
        self.player = VideoPlayer()
        video_col.addWidget(self.player, 1)
        content.addLayout(video_col, 3)

        self.visitors = VisitorsPanel()
        self.visitors.setFixedWidth(320)
        content.addWidget(self.visitors, 0)

        layout.addLayout(content, 1)

        # --- Журнал (таблица последних наблюдений) ---
        journal_title = QLabel("Журнал")
        journal_title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 16px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        layout.addWidget(journal_title)

        self._journal_table = QTableWidget(0, 5)
        self._journal_table.setHorizontalHeaderLabels(
            ["Время", "Клиентов", "Преобладает", "Негатив", "Статус"]
        )
        self._journal_table.verticalHeader().setVisible(False)
        self._journal_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._journal_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._journal_table.setFocusPolicy(Qt.NoFocus)
        self._journal_table.setShowGrid(False)
        self._journal_table.setMinimumHeight(150)
        self._journal_table.setMaximumHeight(230)
        jh = self._journal_table.horizontalHeader()
        jh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        jh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        jh.setSectionResizeMode(2, QHeaderView.Stretch)
        jh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        jh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        layout.addWidget(self._journal_table)

        # --- Сворачиваемая секция «Камера и обработка» ---
        self._details_toggle = QPushButton("Камера и обработка  ⌄")
        self._details_toggle.setObjectName("FlatButton")
        self._details_toggle.setCursor(Qt.PointingHandCursor)
        self._details_toggle.clicked.connect(self._toggle_details)
        layout.addWidget(self._details_toggle)

        self._details_panel = QWidget()
        self._details_panel.setStyleSheet(
            f"background: {T.BG_CARD}; border: 1px solid {T.BORDER};"
            " border-radius: 10px;"
        )
        dl = QVBoxLayout(self._details_panel)
        dl.setContentsMargins(16, 12, 16, 12)
        dl.setSpacing(4)
        self._details_source = QLabel("Источник: —")
        self._details_pipeline = QLabel("Пайплайн: —")
        for lbl in (self._details_source, self._details_pipeline):
            lbl.setStyleSheet(
                f"color: {T.TEXT_SECONDARY}; font-size: 13px;"
                " background: transparent; border: none;"
            )
            dl.addWidget(lbl)
        self._details_panel.setVisible(False)
        layout.addWidget(self._details_panel)

        # Счётчик пригодных кадров (доля кадров с лицами).
        self._frames_total = 0
        self._frames_with_faces = 0

        return panel

    # --- Управление списком потоков ----------------------------------------

    def refresh_stream_list(self) -> None:
        """Перерисовывает список камер, сгруппированных по группам.

        Использует GroupManager для порядка групп, их свёрнутого состояния и
        отображения даже пустых групп. Потоки распределяются по группе
        (StreamConfig.location); без группы — в DEFAULT_GROUP.
        """
        self._clear_list()
        self._camera_widgets.clear()

        runtimes = self.manager.all_runtimes()

        # Группируем потоки по имени группы (location), регистрируя группы.
        by_group: Dict[str, List[StreamRuntime]] = {}
        for rt in runtimes:
            gname = (rt.config.location or "").strip() or DEFAULT_GROUP
            self.group_manager.ensure(gname)
            by_group.setdefault(gname, []).append(rt)

        # Есть ли вообще что показывать (потоки или группы).
        known_groups = [g.name for g in self.group_manager.all()]
        if not runtimes and not known_groups:
            empty = QLabel("Потоков пока нет")
            empty.setStyleSheet(
                f"color: {T.TEXT_MUTED}; padding: 20px; background: transparent;"
                " border: none;"
            )
            empty.setAlignment(Qt.AlignCenter)
            self._list_layout.insertWidget(0, empty)
            self._list_layout.addStretch(1)
            return

        idx = 0
        for gname in known_groups:
            group_streams = by_group.get(gname, [])
            collapsed = self.group_manager.is_collapsed(gname)
            header = _GroupHeader(gname, collapsed, len(group_streams))
            # Клик по заголовку — сворачивание/разворачивание группы.
            header.mousePressEvent = lambda ev, g=gname: self._toggle_group(g)
            # Правый клик — контекстное меню группы.
            header.setContextMenuPolicy(Qt.CustomContextMenu)
            header.customContextMenuRequested.connect(
                lambda pos, g=gname: self._show_group_menu(g)
            )
            self._list_layout.insertWidget(idx, header)
            idx += 1

            if collapsed:
                continue  # Не показываем потоки свёрнутой группы.

            for rt in group_streams:
                item = _CameraItem(rt)
                item.mousePressEvent = lambda ev, sid=rt.config.stream_id: (
                    self.select_stream(sid)
                )
                item.set_status(rt.last_status)
                item.setContextMenuPolicy(Qt.CustomContextMenu)
                item.customContextMenuRequested.connect(
                    lambda pos, sid=rt.config.stream_id: self._show_stream_menu(sid)
                )
                item.menu_requested.connect(
                    lambda sid=rt.config.stream_id, w=item: self._show_stream_menu(
                        sid, w.menu_anchor()
                    )
                )
                self._camera_widgets[rt.config.stream_id] = item
                self._list_layout.insertWidget(idx, item)
                idx += 1

        self._list_layout.addStretch(1)

        # Автовыбор первого потока, если ничего не выбрано.
        if self._selected_id is None and runtimes:
            self.select_stream(runtimes[0].config.stream_id)

    # --- Группы: сворачивание и контекстное меню ---------------------------

    def _toggle_group(self, name: str) -> None:
        """Сворачивает/разворачивает группу и перерисовывает список."""
        self.group_manager.toggle(name)
        self.refresh_stream_list()

    def _show_group_menu(self, name: str) -> None:
        """Контекстное меню группы: добавить поток, переименовать, удалить."""
        self._context_group = name
        menu = QMenu(self)
        menu.setStyleSheet(self._menu_qss())
        act_add = menu.addAction("Добавить поток в группу")
        act_rename = menu.addAction("Переименовать группу")
        act_delete = menu.addAction("Удалить группу")
        chosen = menu.exec(QCursor.pos())
        if chosen == act_add:
            self._on_add_clicked(group=name)
        elif chosen == act_rename:
            self._rename_group(name)
        elif chosen == act_delete:
            self._delete_group(name)

    def _rename_group(self, name: str) -> None:
        """Запрашивает новое имя группы и переименовывает её у всех потоков."""
        new_name, ok = QInputDialog.getText(
            self, "Переименовать группу", "Новое название:", text=name
        )
        new_name = (new_name or "").strip()
        if not ok or not new_name or new_name == name:
            return
        if self.group_manager.exists(new_name):
            msg = f"Группа «{new_name}» уже существует."
            QMessageBox.warning(self, "Группа", msg)
            logger.warning("Группа «%s» уже существует", new_name)
            return
        self.group_manager.rename(name, new_name)
        # Переносим все потоки старой группы в новую.
        for rt in self.manager.all_runtimes():
            if (rt.config.location or "").strip() == name or (
                name == DEFAULT_GROUP and not (rt.config.location or "").strip()
            ):
                rt.config.location = new_name
        self.refresh_stream_list()

    def _delete_group(self, name: str) -> None:
        """Удаляет группу, перенося её потоки в «Без группы»."""
        count = sum(
            1
            for rt in self.manager.all_runtimes()
            if (rt.config.location or "").strip() == name
        )
        msg = f"Удалить группу «{name}»?"
        if count:
            msg += f"\n{count} поток(а) будут перемещены в «{DEFAULT_GROUP}»."
        if QMessageBox.question(self, "Удалить группу", msg) != QMessageBox.Yes:
            return
        self.group_manager.delete(name)
        for rt in self.manager.all_runtimes():
            if (rt.config.location or "").strip() == name:
                rt.config.location = ""  # -> DEFAULT_GROUP
        self.refresh_stream_list()

    @staticmethod
    def _menu_qss() -> str:
        return (
            f"QMenu {{ background: {T.BG_CARD}; color: {T.TEXT_PRIMARY};"
            f" border: 1px solid {T.BORDER}; border-radius: 8px; padding: 4px; }}"
            f"QMenu::item {{ padding: 6px 20px; border-radius: 6px; }}"
            f"QMenu::item:selected {{ background: {T.ACCENT_SOFT}; }}"
        )

    # --- Контекстное меню потока -------------------------------------------

    def _show_stream_menu(
        self, stream_id: str, pos: Optional[QPoint] = None
    ) -> None:
        """Контекстное меню потока: в группу, переименовать, удалить.

        pos — глобальная точка показа (например, под кнопкой «⋯»);
        по умолчанию — позиция курсора (правый клик).
        """
        self._context_stream_id = stream_id
        runtime = self.manager.get_runtime(stream_id)
        if runtime is None:
            return
        menu = QMenu(self)
        menu.setStyleSheet(self._menu_qss())

        move_menu = menu.addMenu("Переместить в группу")
        move_menu.setStyleSheet(self._menu_qss())
        current_group = (runtime.config.location or "").strip() or DEFAULT_GROUP
        group_actions = {}
        for gname in [g.name for g in self.group_manager.all()] or [DEFAULT_GROUP]:
            act = move_menu.addAction(gname)
            # Текущая группа отмечена галочкой.
            act.setCheckable(True)
            act.setChecked(gname == current_group)
            group_actions[act] = gname
        act_new_group = move_menu.addAction("Новая группа…")

        act_rename = menu.addAction("Переименовать поток")
        act_delete = menu.addAction("Удалить поток")

        chosen = menu.exec(pos if pos is not None else QCursor.pos())
        if chosen is None:
            return
        if chosen in group_actions:
            self._move_stream_to_group(stream_id, group_actions[chosen])
        elif chosen == act_new_group:
            self._move_stream_to_new_group(stream_id)
        elif chosen == act_rename:
            self._rename_stream(stream_id)
        elif chosen == act_delete:
            self._delete_stream(stream_id)

    def _move_stream_to_new_group(self, stream_id: str) -> None:
        """Создаёт группу по запросу пользователя и перемещает в неё поток."""
        new_name, ok = QInputDialog.getText(
            self, "Новая группа", "Название новой группы:"
        )
        new_name = (new_name or "").strip()
        if not ok or not new_name:
            return
        self.group_manager.ensure(new_name)
        self._move_stream_to_group(stream_id, new_name)

    def _move_stream_to_group(self, stream_id: str, group_name: str) -> None:
        """Перемещает поток в указанную группу."""
        runtime = self.manager.get_runtime(stream_id)
        if runtime is None:
            return
        self.group_manager.ensure(group_name)
        runtime.config.location = "" if group_name == DEFAULT_GROUP else group_name
        self.refresh_stream_list()

    def _rename_stream(self, stream_id: str) -> None:
        """Переименовывает поток."""
        runtime = self.manager.get_runtime(stream_id)
        if runtime is None:
            return
        new_name, ok = QInputDialog.getText(
            self, "Переименовать поток", "Новое название:", text=runtime.config.name
        )
        new_name = (new_name or "").strip()
        if not ok or not new_name:
            return
        runtime.config.name = new_name
        self.refresh_stream_list()
        if self._selected_id == stream_id:
            self._detail_title.setText(new_name)

    def _delete_stream(self, stream_id: str) -> None:
        """Удаляет поток (с подтверждением)."""
        runtime = self.manager.get_runtime(stream_id)
        if runtime is None:
            return
        if (
            QMessageBox.question(
                self,
                "Удалить поток",
                f"Удалить поток «{runtime.config.name}»?",
            )
            != QMessageBox.Yes
        ):
            return
        was_selected = self._selected_id == stream_id
        if was_selected:
            # Отвязываем плеер только если удаляем именно показываемый поток,
            # иначе «отвалился» бы поток, открытый в детальной панели.
            self._disconnect_video()
            self._selected_id = None
        self.manager.remove_stream(stream_id)
        # При удалённом выбранном потоке refresh автовыберет первый
        # оставшийся; если потоков не осталось — сбрасываем панель.
        self.refresh_stream_list()
        if was_selected and self._selected_id is None:
            self._reset_detail()

    def _reset_detail(self) -> None:
        """Сбрасывает правую панель, когда нечего показывать (нет потоков)."""
        self._detail_title.setText("Выберите поток")
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setText("Включить")
        self._live_badge.set_badge("LIVE", T.STATUS_IDLE)
        self.player.set_placeholder("Кадр видеопотока")
        self.player.set_playing(False)
        self._pending_tracks = []
        self._last_visitors_update = 0.0
        self.visitors.update_tracks([])
        self._clients_metric.set_value("0", "в кадре")
        self._status_metric.set_value("—", "по всем клиентам")
        self._suitable_metric.set_value("—", "за сессию")
        self._frames_total = 0
        self._frames_with_faces = 0
        self._journal_table.setRowCount(0)
        self._incident_banner.set_incident(None)


    def _clear_list(self) -> None:
        # Важно: без setParent(None) — иначе виджет становится top-level
        # окном. takeAt + deleteLater корректно удаляют элемент списка.
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def select_stream(self, stream_id: str) -> None:
        """Делает поток активным: показывает его видео, посетителей, статус."""
        self._selected_id = stream_id
        runtime = self.manager.get_runtime(stream_id)
        if runtime is None:
            return

        # Переподключаем показ кадров к воркеру выбранного потока.
        self._connect_video(runtime)

        self._detail_title.setText(runtime.config.name)
        live = runtime.is_live
        self._live_badge.set_badge(
            "В эфире" if live else "LIVE", T.STATUS_LIVE if live else T.STATUS_IDLE
        )
        self._toggle_btn.setEnabled(True)
        self._toggle_btn.setText("Выключить" if live else "Включить")

        # Заполняем секцию «Камера и обработка» для выбранного потока.
        self._details_source.setText(f"Источник: {runtime.config.capture_source}")
        self._current_pipeline_label = runtime.config.pipeline_key
        self._details_pipeline.setText(
            f"Пайплайн: {runtime.config.pipeline_key} · FPS: —"
        )

        if not live:
            self.player.set_placeholder("Кадр видеопотока")
            self.player.set_playing(False)
            self._pending_tracks = []
            self._last_visitors_update = 0.0
            self.visitors.update_tracks([])
            self._clients_metric.set_value("0", "в кадре")
            self._status_metric.set_value("—", "по всем клиентам")
            self._suitable_metric.set_value("—", "за сессию")
            self._frames_total = 0
            self._frames_with_faces = 0
            self._journal_table.setRowCount(0)
            self._incident_banner.set_incident(None)

        # Подсветка выбранной карточки.
        for sid, widget in self._camera_widgets.items():
            selected = sid == stream_id
            widget.setStyleSheet(
                f"background: {T.BG_CARD_HOVER if selected else T.BG_CARD};"
                f" border: 1px solid {T.ACCENT if selected else T.BORDER};"
                " border-radius: 10px;"
            )

    def _connect_video(self, runtime: StreamRuntime) -> None:
        """Подключает сигналы активного воркера к плееру и наоборот.

        При переключении камеры отключаем предыдущий воркер, чтобы его кадры
        не «утекали» в плеер другой камеры.
        """
        # Отключаем предыдущего воркера, если он был.
        self._disconnect_video()

        worker = runtime.worker
        if worker is None:
            return

        # Воркер -> плеер / UI
        worker.frame_ready.connect(self.player.update_frame)
        worker.fps_updated.connect(self._on_fps_updated)
        worker.position_changed.connect(self.player.set_position)
        worker.media_info.connect(self.player.set_media_info)
        worker.paused_changed.connect(self.player.set_playing)
        # Когда поток стартовал — плеер в состоянии «играет».
        worker.media_info.connect(
            lambda fps, dur, seek: self.player.set_playing(True)
        )

        # Плеер -> воркер (намерения пользователя)
        self.player.play_pause_clicked.connect(worker.toggle_pause)
        self.player.seek_forward_clicked.connect(
            lambda: worker.seek_by(VideoPlayer.SEEK_STEP_SECONDS)
        )
        self.player.seek_backward_clicked.connect(
            lambda: worker.seek_by(-VideoPlayer.SEEK_STEP_SECONDS)
        )
        self.player.seek_to_ratio.connect(
            lambda ratio: self._seek_worker_to_ratio(worker, ratio)
        )

        self._frame_conns["current"] = worker

        # Обновляем секцию «Камера и обработка».
        self._details_source.setText(f"Источник: {runtime.config.capture_source}")
        self._current_pipeline_label = runtime.config.pipeline_key
        self._details_pipeline.setText(
            f"Пайплайн: {runtime.config.pipeline_key} · FPS: —"
        )

    def _disconnect_video(self) -> None:
        """Отключает сигналы ранее активного воркера от плеера."""
        worker = self._frame_conns.get("current")
        if worker is None:
            return
        try:
            worker.frame_ready.disconnect(self.player.update_frame)
            worker.fps_updated.disconnect(self._on_fps_updated)
            worker.position_changed.disconnect(self.player.set_position)
            worker.media_info.disconnect(self.player.set_media_info)
            worker.paused_changed.disconnect(self.player.set_playing)
        except (RuntimeError, TypeError):
            # Связи могло не быть — игнорируем.
            pass
        self._frame_conns["current"] = None

    @staticmethod
    def _seek_worker_to_ratio(worker, ratio: float) -> None:
        """Перематывает воркер на позицию ratio (0..1) от его длительности."""
        duration = getattr(worker, "_duration", 0.0) or 0.0
        if duration <= 0 or not worker.seekable:
            return
        target_seconds = max(0.0, min(1.0, ratio)) * duration
        current = getattr(worker, "_current_position_seconds", lambda: 0.0)()
        # Перемотка относительная: сдвигаемся от текущей позиции к цели.
        worker.seek_by(target_seconds - current)


    def _on_fps_updated(self, fps: float) -> None:
        """Обновляет метрику FPS для выбранного потока."""
        self._suitable_metric.set_value(f"{fps:.1f}", FPS_CAPTION)


    def _on_add_clicked(self, group: Optional[str] = None) -> None:
        """Открывает диалог добавления потока и регистрирует его.

        Если задана группа — она будет предвыбрана в диалоге.
        """
        from app.ui.dialogs.add_stream_dialog import AddStreamDialog

        groups = [g.name for g in self.group_manager.all()] or [DEFAULT_GROUP]
        dialog = AddStreamDialog(
            self,
            groups=groups,
            default_group=group,
            preset_manager=self.preset_manager,
        )
        if dialog.exec():
            config = dialog.result_config()
            if config is not None:
                self.group_manager.ensure(config.location or DEFAULT_GROUP)
                self.manager.add_stream(config)
                self.refresh_stream_list()
                self.select_stream(config.stream_id)

    def _on_toggle_clicked(self) -> None:
        """Включает/выключает выбранный поток и переподключает показ кадров."""
        if not self._selected_id:
            return
        self.manager.toggle_stream(self._selected_id)
        # После старта создаётся новый воркер — переподключаем сигналы кадров.
        runtime = self.manager.get_runtime(self._selected_id)
        if runtime is not None and runtime.is_live:
            self._connect_video(runtime)
            self._toggle_btn.setText("Выключить")
            self._live_badge.set_badge("В эфире", T.STATUS_LIVE)
            self.player.set_playing(True)
        elif runtime is not None:
            self.player.set_placeholder("Нет сигнала")
            self.player.set_playing(False)
            self._toggle_btn.setText("Включить")
            self._live_badge.set_badge("LIVE", T.STATUS_IDLE)

    # --- Обработчики сигналов менеджера ------------------------------------

    def _on_status_changed(self, stream_id: str, status: str) -> None:
        """Обновляет статус в карточке камеры и в детальной панели."""
        widget = self._camera_widgets.get(stream_id)
        if widget is not None:
            widget.set_status(status)

        if stream_id == self._selected_id:
            runtime = self.manager.get_runtime(stream_id)
            live = runtime.is_live if runtime else False
            self._toggle_btn.setText("Выключить" if live else "Включить")
            self._live_badge.set_badge(
                "В эфире" if live else "LIVE",
                T.STATUS_LIVE if live else T.STATUS_IDLE,
            )
            if not live:
                self.player.set_placeholder("Нет сигнала")

    def _on_tracks_updated(self, stream_id: str, tracks: list) -> None:
        """Обновляет панель посетителей и метрики.

        Сигнал приходит на КАЖДОМ кадре (десятки раз в секунду). Метрики
        лёгкие — обновляем сразу. А вот панель «Посетители»
        перерисовывает десятки виджетов, поэтому троттлим её (~5 раз/сек),
        чтобы не грузить UI и не пытаться «успеть» за каждым кадром.
        """
        if stream_id != self._selected_id:
            # Для фоновых потоков обновляем только бейдж на карточке.
            summary = StreamManager.summarize_tracks(tracks)
            widget = self._camera_widgets.get(stream_id)
            if widget is not None:
                widget.set_badge(*self._badge_for_summary(summary))
            return

        # Кэшируем последние треки выбранного потока.
        self._pending_tracks = list(tracks)
        summary = StreamManager.summarize_tracks(tracks)
        self._clients_metric.set_value(str(len(tracks)), "в кадре")

        incident_count = summary.get(STATUS_INCIDENT, 0)
        if incident_count > 0:
            self._status_metric.set_value(
                "Инцидент", T.STATUS_INCIDENT
            )
            self._incident_banner.set_incident(
                f"Клиентов с инцидентом: {incident_count}"
            )
            self._add_journal_row(
                len(tracks), "Злость/Грусть/Отвращение", "Инцидент", T.STATUS_INCIDENT
            )
        elif summary.get(STATUS_UNSURE, 0) > 0:
            self._status_metric.set_value("Не уверена", T.STATUS_WARN)
            self._incident_banner.set_incident(None)
        elif summary.get(STATUS_NORMAL, 0) > 0:
            self._status_metric.set_value("Норма", T.STATUS_LIVE)
            self._incident_banner.set_incident(None)
        else:
            self._status_metric.set_value("—", T.TEXT_SECONDARY)
            self._incident_banner.set_incident(None)

        # Троттлинг тяжёлой панели посетителей.
        self._maybe_refresh_visitors()

    def _add_journal_row(
        self,
        clients: int,
        emotion: str,
        status: str,
        status_color: str,
    ) -> None:
        """Добавляет строку в журнал наблюдений."""
        from datetime import datetime

        now = datetime.now().strftime("%H:%M:%S")
        table = self._journal_table
        row = table.rowCount()
        table.insertRow(row)
        items = [
            QTableWidgetItem(now),
            QTableWidgetItem(str(clients)),
            QTableWidgetItem(emotion),
            QTableWidgetItem("—"),
            QTableWidgetItem(status),
        ]
        items[4].setForeground(QColor(status_color))
        for col, item in enumerate(items):
            table.setItem(row, col, item)
        # Оставляем не более 15 строк.
        while table.rowCount() > 15:
            table.removeRow(0)
        table.scrollToBottom()

    def _maybe_refresh_visitors(self) -> None:
        """Перерисовывает панель посетителей не чаще, чем раз в интервал."""
        now = time.monotonic()
        if now - self._last_visitors_update < VISITORS_REFRESH_SECONDS:
            return
        self._last_visitors_update = now
        self.visitors.update_tracks(self._pending_tracks)

    @staticmethod
    def _badge_for_summary(summary: Dict[str, int]):
        """Возвращает (текст, цвет) бейджа по сводке статусов клиентов."""
        if summary.get(STATUS_INCIDENT, 0) > 0:
            return "Инцидент", T.STATUS_INCIDENT
        if summary.get(STATUS_UNSURE, 0) > 0:
            return "Не уверена", T.STATUS_WARN
        if summary.get(STATUS_NORMAL, 0) > 0:
            return "Норма", T.STATUS_LIVE
        return "—", T.TEXT_MUTED

    # --- Обработчики баннера инцидента --------------------------------------

    def _on_incident_confirm(self) -> None:
        """Пользователь подтвердил инцидент — скрывает баннер."""
        self._incident_banner.set_incident(None)

    def _on_incident_false(self) -> None:
        """Пользователь пометил инцидент как ложное срабатывание."""
        self._incident_banner.set_incident(None)

    def _on_incident_dismiss(self) -> None:
        """Пользователь скрыл баннер."""
        self._incident_banner.set_incident(None)

    def _toggle_details(self) -> None:
        """Разворачивает/сворачивает секцию «Камера и обработка»."""
        visible = not self._details_panel.isVisible()
        self._details_panel.setVisible(visible)
        self._details_toggle.setText(
            f"Камера и обработка {'⌃' if visible else '⌄'}"
        )


