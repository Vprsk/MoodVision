"""Offscreen-проверка страницы «Потоки»: меню потока, группы, удаление.

Запуск:  QT_QPA_PLATFORM=offscreen python scripts/streams_ui_check.py
Проверяет: убраны вкладки «Анализ видео»/«Настройки» и баннер инцидента,
кнопка «⋯» на карточке, меню потока (переместить в группу / новая группа /
удалить), автовыбор и сброс панели при удалении.
Завершается автоматически, код возврата 0 = OK.
"""

import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

FAILURES: list = []


def check(name: str, cond: bool, extra: str = "") -> None:
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILURES.append(name)


# --- Фейки вместо модальных диалогов/меню -----------------------------------

class _Act:
    """Мини-замена QAction для фейкового меню."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._checkable = False
        self._checked = False

    def text(self) -> str:
        return self._text

    def setCheckable(self, v: bool) -> None:  # noqa: N802 (Qt API)
        self._checkable = v

    def setChecked(self, v: bool) -> None:  # noqa: N802 (Qt API)
        self._checked = v

    def isChecked(self) -> bool:  # noqa: N802 (Qt API)
        return self._checked


class _Menu:
    """Фейковый QMenu: строит те же пункты, но exec() возвращает pick."""

    pick: str | None = None  # текст пункта, который «выберет» пользователь

    def __init__(self, parent=None) -> None:
        self._items: list = []
        self._title = ""

    def setStyleSheet(self, *_a) -> None:  # noqa: N802 (Qt API)
        pass

    def addMenu(self, title: str):  # noqa: N802 (Qt API)
        sub = _Menu()
        sub._title = title
        self._items.append(sub)
        return sub

    def addAction(self, text: str):  # noqa: N802 (Qt API)
        act = _Act(text)
        self._items.append(act)
        return act

    def exec(self, pos=None, *_a, **_k):  # noqa: N802 (Qt API)
        return self._find(_Menu.pick)

    def _find(self, text):
        for item in self._items:
            if isinstance(item, _Act) and item.text() == text:
                return item
            if isinstance(item, _Menu):
                got = item._find(text)
                if got is not None:
                    return got
        return None


class _FakeMessageBox:
    Yes = 1

    @staticmethod
    def question(*_a, **_k) -> int:
        return _FakeMessageBox.Yes


class _FakeInputDialog:
    @staticmethod
    def getText(*_a, **_k):  # noqa: N802 (Qt API)
        return "Архив", True


def main() -> int:
    app = QApplication(sys.argv)  # noqa: F841 — Qt-приложение обязательно

    import app.ui.streams_page as sp
    from app.core.group_manager import GroupManager
    from app.core.stream_config import StreamConfig
    from app.core.stream_manager import StreamManager
    from app.ui.top_bar import MENU_ITEMS

    print("== Навигация: убраны «Анализ видео» и «Настройки» ==")
    keys = [k for k, _ in MENU_ITEMS]
    check("пункты меню = Потоки/История/Пресеты",
          keys == ["streams", "history", "presets"], str(keys))

    print("== Модуль streams_page: баннер инцидента убран ==")
    check("QCursor импортирован (иначе меню падало бы)", "QCursor" in vars(sp))
    check("IncidentBanner не импортируется", "IncidentBanner" not in vars(sp))
    check("Signal импортирован для кнопки «⋯»", "Signal" in vars(sp))

    print("== StreamsPage: карточки, кнопка «⋯», меню ==")
    tmpdir = Path(tempfile.mkdtemp(prefix="streams-ui-check-"))
    gm = GroupManager(path=tmpdir / "groups.json")
    sm = StreamManager(None)
    page = sp.StreamsPage(sm, gm, None)

    cfg_a = StreamConfig(name="Камера A", source="webcam:0", location="")
    cfg_b = StreamConfig(name="Камера B", source="webcam:1", location="Склад")
    cfg_c = StreamConfig(name="Камера C", source="webcam:2", location="Кабинет")
    for cfg in (cfg_a, cfg_b, cfg_c):
        sm.add_stream(cfg)
    page.refresh_stream_list()

    check("баннер отсутствует на странице", not hasattr(page, "banner"))
    check("3 карточки в списке", len(page._camera_widgets) == 3,
          str(len(page._camera_widgets)))
    w_a = page._camera_widgets[cfg_a.stream_id]
    check("кнопка «⋯» на карточке", w_a._menu_btn.text() == "⋯")

    # Клик по «⋯» должен открывать _show_stream_menu с точкой привязки.
    calls: list = []
    orig_show = page._show_stream_menu
    page._show_stream_menu = lambda sid, pos=None: calls.append((sid, pos))
    w_a.menu_requested.emit()
    page._show_stream_menu = orig_show
    check("«⋯» вызывает меню с anchor-позицией",
          len(calls) == 1 and calls[0][0] == cfg_a.stream_id
          and isinstance(calls[0][1], QPoint),
          str(calls))

    # Подменяем диалоги/меню на неблокирующие фейки.
    real_qmenu, real_msgbox, real_input = sp.QMenu, sp.QMessageBox, sp.QInputDialog
    sp.QMenu, sp.QMessageBox, sp.QInputDialog = _Menu, _FakeMessageBox, _FakeInputDialog
    try:
        # Правый клик: _show_stream_menu без pos считает QCursor.pos()
        # (в старой версии здесь был NameError — QCursor не был импортирован).
        page.select_stream(cfg_a.stream_id)
        disc = {"n": 0}
        orig_disc = page._disconnect_video

        def counting_disc() -> None:
            disc["n"] += 1
            orig_disc()

        page._disconnect_video = counting_disc

        # --- Удаление фонового потока через меню ---
        _Menu.pick = "Удалить поток"
        page._show_stream_menu(cfg_c.stream_id)
        check("удаление фонового потока через меню",
              sm.get_runtime(cfg_c.stream_id) is None)
        check("фоновое удаление не роняет показ выбранного",
              disc["n"] == 0 and page._selected_id == cfg_a.stream_id,
              f"disconnect={disc['n']}, selected={page._selected_id}")

        # --- Перемещение между группами через меню ---
        _Menu.pick = "Склад"
        page._show_stream_menu(cfg_a.stream_id)
        check("переместил A в «Склад» через меню",
              cfg_a.location == "Склад", cfg_a.location)
        check("карточки перегруппированы",
              cfg_a.stream_id in page._camera_widgets and gm.exists("Склад"))

        # --- Новая группа прямо из меню перемещения ---
        _Menu.pick = "Новая группа…"
        page._show_stream_menu(cfg_a.stream_id)
        check("переместил A в новую группу «Архив»",
              cfg_a.location == "Архив" and gm.exists("Архив"),
              cfg_a.location)

        # --- Удаление выбранного потока через меню ---
        _Menu.pick = "Удалить поток"
        page._show_stream_menu(cfg_a.stream_id)
        check("удаление выбранного потока через меню",
              sm.get_runtime(cfg_a.stream_id) is None)
        check("после удаления выбранного — автовыбор остатка",
              page._selected_id == cfg_b.stream_id,
              str(page._selected_id))
        check("при удалении выбранного: отвязка на удаление + автовыбор",
              disc["n"] == 2,
              f"disconnect={disc['n']} (1 удаление + 1 переподключение)")

        # --- Удаление последнего потока сбрасывает панель ---
        _Menu.pick = "Удалить поток"
        page._show_stream_menu(cfg_b.stream_id)
        check("последний поток удалён", sm.all_runtimes() == [])
        check("панель сброшена",
              page._detail_title.text() == "Выберите поток"
              and not page._toggle_btn.isEnabled())
        check("в списке пустое состояние", len(page._camera_widgets) == 0)

        page._disconnect_video = orig_disc
    finally:
        sp.QMenu, sp.QMessageBox, sp.QInputDialog = (
            real_qmenu, real_msgbox, real_input
        )

    print("== MainWindow: 3 вкладки ==")
    from app.ui.main_window import MainWindow

    win = MainWindow()
    check("вкладок 3 (Потоки/История/Пресеты)", win.pages.count() == 3,
          f"сейчас {win.pages.count()}")
    check("нет analysis_page / settings_page",
          not hasattr(win, "analysis_page") and not hasattr(win, "settings_page"))
    win._navigate("presets")
    check("навигация на Пресеты", win.pages.currentWidget() is win.presets_page)
    try:
        win._navigate("settings")   # удалённый ключ не должен ронять
        win._navigate("analysis")
        check("навигация по удалённым ключам не падает", True)
    except Exception as exc:  # noqa: BLE001
        check("навигация по удалённым ключам не падает", False, repr(exc))
    win.close()
    app.processEvents()

    print("\n=== ИТОГ ===")
    if FAILURES:
        print(f"Провалено: {len(FAILURES)} → {FAILURES}")
        print("РЕЗУЛЬТАТ: FAIL")
        return 1
    print("РЕЗУЛЬТАТ: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — показать полный traceback падения
        traceback.print_exc()
        raise SystemExit(1)

