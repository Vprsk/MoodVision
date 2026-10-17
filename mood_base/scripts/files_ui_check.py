"""Offscreen-проверка вкладок «Файлы» и «История» без запуска модели.

Запуск:  QT_QPA_PLATFORM=offscreen python mood_base/scripts/files_ui_check.py
(из корня проекта; скрипт сам добавляет корень в sys.path).

Проверяет:
  * интеграцию вкладок в MainWindow — 4 страницы в стеке, порядок совпадает
    с MENU_ITEMS (streams->0, files->1, history->2, presets->3), навигация
    через top_bar.navigate;
  * очередь FilesPage — добавление файлов (эмуляция drag-and-drop) и
    удаление без запуска воркеров;
  * карточки HistoryPage из временных фикстур хранилища (save_log в
    data/analysis/): количество карточек, заголовок с basename, клик по
    шапке, таблица детекций, аккордеон, пустой placeholder;
  * сигнал analysis_completed → refresh() истории;
  * shutdown: closeEvent без ошибок.

Модель и реальный анализ НЕ запускаются. Временные папки
data/analysis/<...> удаляются в finally. Завершается автоматически,
код возврата 0 = OK.
"""

import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Как в streams_ui_check.py: добавляем mood_base/ (родитель scripts/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Корень проекта (родитель mood_base) — для импортов app.*/contracts.*.
APP_ROOT = PROJECT_ROOT.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from PySide6.QtCore import QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QTableWidget  # noqa: E402

FAILURES: list = []


def check(name: str, cond: bool, extra: str = "") -> None:
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILURES.append(name)


def _card_has_basename(card, source: str) -> bool:
    """Есть ли в заголовке карточки имя файла (basename source)."""
    base = os.path.basename(source)
    return any(base in label.text() for label in card.findChildren(QLabel))


def main() -> int:
    app = QApplication(sys.argv)  # noqa: F841 — Qt-приложение обязательно

    from app.core.file_analysis import analysis_dir_for, list_analyses, save_log
    from app.ui.files_page import FilesPage
    from app.ui.main_window import MainWindow
    from app.ui.top_bar import MENU_ITEMS
    import app.ui.history_page as hp

    # Runtime-файлы, которые closeEvent может создать (state/, presets.json).
    state_streams = APP_ROOT / "state" / "streams.json"
    state_groups = APP_ROOT / "state" / "groups.json"
    project_presets = APP_ROOT / "presets.json"
    existed = {p: p.exists() for p in (state_streams, state_groups, project_presets)}

    print("== MainWindow: интеграция вкладок ==")
    win = MainWindow()
    keys = [k for k, _ in MENU_ITEMS]
    check("пункты меню = Потоки/Файлы/История/Пресеты",
          keys == ["streams", "files", "history", "presets"], str(keys))
    check("в стеке 4 страницы", win.pages.count() == 4, f"сейчас {win.pages.count()}")
    check("порядок страниц = MENU_ITEMS",
          [win.pages.widget(i) for i in range(win.pages.count())]
          == [win.streams_page, win.files_page, win.history_page, win.presets_page])
    check("_page_index streams->0 files->1 history->2 presets->3",
          win._page_index == {"streams": 0, "files": 1, "history": 2, "presets": 3},
          str(win._page_index))
    win.top_bar.navigate.emit("files")
    check("навигация на «Файлы»", win.pages.currentWidget() is win.files_page)
    win.top_bar.navigate.emit("history")
    check("навигация на «Историю»", win.pages.currentWidget() is win.history_page)

    print("== FilesPage: очередь ==")
    page = win.files_page
    # Автозапуск мешает проверить саму очередь (воркеры сразу бы стартовали
    # на фейковых путях) — выключаем и включаем обратно после проверок.
    page._auto_start = False
    ts = int(time.time() * 1000)
    p1 = f"C:/fake/videos/files_check_{ts}_1.mp4"
    p2 = f"C:/fake/videos/files_check_{ts}_2.mp4"
    page.drop.files_dropped([p1, p2])  # эмуляция drag-and-drop из проводника
    check("в очереди 2 файла",
          page._queue == [p1, p2] and page.queue_list.count() == 2, str(page._queue))
    check("кнопка «Анализировать» активна", page._analyze_btn.isEnabled())
    check("воркеры не запущены",
          page._workers == [] and page._current_worker is None and not page._analyzing)
    page.queue_list.setCurrentRow(0)
    page._on_remove_selected()
    check("удалён выбранный файл",
          page._queue == [p2] and page.queue_list.count() == 1, str(page._queue))
    page._on_clear_queue()
    check("очередь очищена", page._queue == [] and page.queue_list.count() == 0)
    check("кнопка «Анализировать» выключена", not page._analyze_btn.isEnabled())
    # Возвращаем автозапуск в исходное (рабочее) состояние.
    page._auto_start = True

    print("== HistoryPage: карточки из хранилища ==")
    fixture_dirs: list = []
    try:
        # --- Временные фикстуры: 2 анализа в data/analysis/ ---
        now = datetime.now()
        dir_a = analysis_dir_for(p1, when=now)
        dir_b = analysis_dir_for(p2, when=now + timedelta(seconds=1))
        fixture_dirs = [dir_a, dir_b]

        log_a = {
            "source_file": p1,
            "created_at": "2026-10-17T12:00:00+05:00",
            "video": {"fps": 25.0, "width": 640, "height": 480,
                      "total_frames": 2, "duration_sec": 0.08},
            "results": [
                {"frame_id": 1, "detections": [
                    {"track_id": 1, "bbox": [10, 20, 30, 40], "emotion": "happy",
                     "face_confidence": 0.95, "emotion_confidence": 0.87},
                ]},
                {"frame_id": 2, "detections": [
                    {"track_id": 1, "bbox": [12, 22, 32, 42], "emotion": "neutral",
                     "face_confidence": 0.90, "emotion_confidence": 0.60},
                    {"track_id": 2, "bbox": [100, 200, 150, 250], "emotion": "sad",
                     "face_confidence": 0.80, "emotion_confidence": 0.70},
                ]},
            ],
        }
        log_b = {
            "source_file": p2,
            "created_at": "2026-10-17T13:00:00+05:00",
            "video": {"fps": 30.0, "width": 1280, "height": 720,
                      "total_frames": 1, "duration_sec": 0.03},
            "results": [
                {"frame_id": 1, "detections": [
                    {"track_id": 3, "bbox": [5, 5, 25, 25], "emotion": "angry",
                     "face_confidence": 0.70, "emotion_confidence": 0.55},
                ]},
            ],
        }
        save_log(dir_a, log_a)
        save_log(dir_b, log_b)

        found = {a["analysis_id"] for a in list_analyses()}
        check("list_analyses находит обе фикстуры",
              {dir_a.name, dir_b.name} <= found, str(sorted(found)))

        # Изолируем проверку UI от реального хранилища: list_analyses в модуле
        # history_page временно фильтрует только наши фикстуры (формат элементов
        # — реальный, load_analysis читает настоящие логи из data/analysis/).
        real_list = hp.list_analyses
        fixture_ids = {dir_a.name, dir_b.name}
        hp.list_analyses = lambda: [a for a in real_list()
                                    if a["analysis_id"] in fixture_ids]
        try:
            hp_page = win.history_page

            hp.list_analyses = lambda: []
            hp_page.refresh()
            check("placeholder виден без анализов",
                  not hp_page._empty.isHidden() and hp_page.scroll.isHidden())

            hp.list_analyses = lambda: [a for a in real_list()
                                        if a["analysis_id"] in fixture_ids]
            hp_page.refresh()
            check("карточек 2", len(hp_page._cards) == 2, str(len(hp_page._cards)))
            check("placeholder скрыт при карточках",
                  hp_page._empty.isHidden() and not hp_page.scroll.isHidden())

            # Сортировка по created_at (убывание): B (13:00) раньше A (12:00).
            card_b, card_a = hp_page._cards
            check("заголовок карточки B содержит basename",
                  _card_has_basename(card_b, p2))
            check("заголовок карточки A содержит basename",
                  _card_has_basename(card_a, p1))

            # Клик по шапке разворачивает карточку (аккордеон).
            header_a = card_a.findChild(hp._CardHeader)
            header_a.clicked.emit()
            check("клик по шапке разворачивает карточку",
                  card_a.is_expanded and not card_a._content.isHidden())

            table = card_a.findChild(QTableWidget)
            check("таблица детекций построена", table is not None)
            if table is not None:
                check("строк таблицы 3", table.rowCount() == 3, str(table.rowCount()))
                check("ячейка frame_id", table.item(0, 0).text() == "1",
                      table.item(0, 0).text())
                check("ячейка bbox", table.item(0, 1).text() == "[10, 20, 30, 40]",
                      table.item(0, 1).text())
                check("ячейка emotion", table.item(0, 2).text() == "happy",
                      table.item(0, 2).text())
                check("ячейка уверенность", table.item(0, 3).text() == "0.87",
                      table.item(0, 3).text())
                check("второй кадр: neutral/sad",
                      table.item(1, 2).text() == "neutral"
                      and table.item(2, 2).text() == "sad",
                      f"{table.item(1, 2).text()}/{table.item(2, 2).text()}")

            # Открытие второй карточки сворачивает первую.
            header_b = card_b.findChild(hp._CardHeader)
            header_b.clicked.emit()
            check("аккордеон: B открыта, A свернута",
                  card_b.is_expanded and not card_a.is_expanded
                  and hp_page._expanded_id == card_b.analysis_id)
        finally:
            hp.list_analyses = real_list
    finally:
        for d in fixture_dirs:
            shutil.rmtree(d, ignore_errors=True)

    print("== Сигнал обновления истории ==")
    check("FilesPage.analysis_completed — Signal",
          isinstance(FilesPage.analysis_completed, Signal))
    calls = {"n": 0}

    def _count_refresh() -> None:
        calls["n"] += 1

    orig_refresh = win.history_page.refresh
    win.history_page.refresh = _count_refresh
    try:
        win._on_analysis_completed("fake_dir")
        check("_on_analysis_completed вызывает refresh()", calls["n"] == 1,
              f"вызовов {calls['n']}")
        win.files_page.analysis_completed.emit("fake_dir")
        check("сигнал analysis_completed -> refresh()", calls["n"] == 2,
              f"вызовов {calls['n']}")
    finally:
        win.history_page.refresh = orig_refresh

    print("== Shutdown ==")
    try:
        win.close()
        check("closeEvent без ошибок", True)
    except Exception as exc:  # noqa: BLE001
        check("closeEvent без ошибок", False, repr(exc))
    app.processEvents()

    # Убираем runtime-файлы, созданные closeEvent (если их не было до теста).
    for p, was in existed.items():
        if not was and p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    # Автофиниш: обрабатываем отложенные события (deleteLater и т.п.).
    QTimer.singleShot(0, app.quit)
    app.exec()

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
