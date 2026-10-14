"""Главное окно приложения «EmoSense».

Связывает верхнюю панель навигации, менеджер потоков и страницы
(QStackedWidget). Отвечает за жизненный цикл приложения и корректную
остановку всех воркеров при закрытии.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.group_manager import GroupManager
from app.core.persistence import load_streams, save_streams
from app.core.preset_manager import PresetManager
from app.core.stream_manager import StreamManager
from app.ui.streams_page import StreamsPage
from app.ui.top_bar import TopBar


class MainWindow(QMainWindow):
    """Основное окно приложения."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EmoSense — анализ эмоций видеопотока")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)

        # --- Данные / логика ---
        self.groups = GroupManager()
        self.presets = PresetManager()
        self.manager = StreamManager(self)

        # --- Корневой layout ---
        root = QWidget()
        root.setObjectName("RootWidget")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.top_bar = TopBar()
        layout.addWidget(self.top_bar)

        self.pages = QStackedWidget()
        self.pages.setObjectName("RootWidget")

        # Создаём страницы. Импорт выполняется здесь, чтобы избежать
        # циклической загрузки модулей на уровне пакета.
        from app.ui.history_page import HistoryPage
        from app.ui.presets_page import PresetsPage

        self.streams_page = StreamsPage(
            self.manager, self.groups, self.presets, self
        )
        self.history_page = HistoryPage(self)
        self.presets_page = PresetsPage(self.presets, self)

        self.pages.addWidget(self.streams_page)    # index 0
        self.pages.addWidget(self.history_page)    # index 1
        self.pages.addWidget(self.presets_page)    # index 2

        layout.addWidget(self.pages, 1)
        self.setCentralWidget(root)

        # --- Навигация ---
        self.top_bar.navigate.connect(self._navigate)
        self._navigate("streams")

        # --- Восстановление сохранённых потоков ---
        # Выполняется после построения streams_page, чтобы список и выбор
        # первого потока корректно отрисовались с восстановленными данными.
        self._load_streams()
        self.streams_page.refresh_stream_list()

    def _navigate(self, key: str) -> None:
        """Переключает страницу по ключу меню."""
        index = {
            "streams": 0,
            "history": 1,
            "presets": 2,
        }.get(key, 0)
        self.pages.setCurrentIndex(index)
        self.top_bar.set_active(key)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Сохраняет конфиги потоков и пресеты, останавливает воркеры."""
        self._save_streams()
        try:
            self.presets.save()
        except Exception as exc:  # noqa: BLE001 — не роняем закрытие из-за записи
            print(f"Не удалось сохранить пресеты: {exc}")
        self.manager.stop_all()
        super().closeEvent(event)

    def _save_streams(self) -> None:
        """Сохраняет конфиги потоков и список групп на диск.

        Флаг enabled синхронизируется с фактическим live-статусом, чтобы при
        следующем запуске корректно автозапустить нужные потоки.
        """
        configs = []
        for rt in self.manager.all_runtimes():
            rt.config.enabled = rt.is_live
            configs.append(rt.config)
        try:
            save_streams(configs)
            self.groups.save()
        except Exception as exc:  # noqa: BLE001 — не роняем закрытие из-за записи
            print(f"Не удалось сохранить состояние: {exc}")

    def _load_streams(self) -> None:
        """Восстанавливает потоки из файла и запускает ранее активные."""
        try:
            configs = load_streams()
        except Exception as exc:  # noqa: BLE001
            print(f"Не удалось загрузить потоки: {exc}")
            return
        for config in configs:
            # Регистрируем группу потока, чтобы она попалась в список групп.
            self.groups.ensure(config.location or GroupManager.DEFAULT_GROUP)
            self.manager.add_stream(config)
            if config.enabled:
                self.manager.start_stream(config.stream_id)
