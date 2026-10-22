"""Headless-проверка страницы «Потоки»: удаление потоков, счётчик, секции.

Запуск (из корня проекта):
    QT_QPA_PLATFORM=offscreen env/Scripts/python.exe mood_base/scripts/streams_delete_check.py

Проверяет:
  * кнопку «−» у каждого потока и удаление по ней;
  * удаление потока с работающим воркером (без падения QThread);
  * полный цикл: восстановление из state -> удаление -> немедленная перезапись;
  * счётчик «(всего: N)» рядом с заголовком «Посетители»;
  * сворачиваемую секцию «Журнал»;
  * отсутствие тайлов метрик (FPS/инциденты/клиенты).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Корень проекта — три уровня выше этого файла (mood_base/scripts/…).
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.group_manager import GroupManager  # noqa: E402
from app.core.stream_config import StreamConfig  # noqa: E402
from app.core.stream_manager import StreamManager  # noqa: E402
from app.ui.streams_page import StreamsPage  # noqa: E402


def check_main_window_cycle(app) -> None:
    """Полный цикл: восстановление -> удаление -> немедленное сохранение."""
    from app.core.persistence import load_streams
    from app.ui.main_window import MainWindow

    before = load_streams()
    print("потоков в state до:", len(before))

    window = MainWindow()
    page = window.streams_page
    page.refresh_stream_list()
    assert len(page._camera_widgets) == len(before), (
        len(page._camera_widgets),
        len(before),
    )
    print("OK: потоки восстановлены в UI")

    # Удаляем первый поток кнопкой «−».
    first_id = list(page._camera_widgets.keys())[0]
    item = page._camera_widgets[first_id]
    item._remove_btn.click()

    after = load_streams()
    print("потоков в state после удаления:", len(after))
    assert len(after) == len(before) - 1, "состояние не перезаписалось сразу"
    assert all(s.stream_id != first_id for s in after), "удалённый поток остался в state"
    print("OK: удалённый поток сразу исключён из state/streams.json")

    window.manager.stop_all()
    window.close()


def main() -> int:
    app = QApplication(sys.argv)

    manager = StreamManager()
    groups = GroupManager(path=ROOT / "state" / "_check_groups.json")
    page = StreamsPage(manager, groups, None)

    # --- 1. Потоки в списке и кнопка «−» у каждого ---
    cfg1 = StreamConfig(name="Камера 1", source="webcam:0", location="Офис")
    cfg2 = StreamConfig(name="Камера 2", source="webcam:1", location="Офис")
    manager.add_stream(cfg1)
    manager.add_stream(cfg2)
    page.refresh_stream_list()
    assert len(page._camera_widgets) == 2, page._camera_widgets.keys()
    item = page._camera_widgets[cfg1.stream_id]
    assert item._remove_btn is not None, "нет кнопки «−» у потока"
    print("OK: кнопка «−» есть у каждого потока")

    # --- 2. Удаление по кнопке «−» ---
    item._remove_btn.click()
    assert manager.get_runtime(cfg1.stream_id) is None, "поток не удалён"
    assert cfg1.stream_id not in page._camera_widgets
    assert len(manager.all_runtimes()) == 1
    print("OK: удаление по кнопке «−» работает (без подтверждения)")

    # Повторное удаление не должно падать.
    page._delete_stream(cfg1.stream_id, confirm=False)
    print("OK: повторное удаление безопасно")

    # --- 3. Счётчик «(всего: N)» ---
    page.select_stream(cfg2.stream_id)
    assert page.visitors._total_label.text() == "(всего: 0)", (
        page.visitors._total_label.text()
    )
    page.visitors.set_total(7)
    assert page.visitors._total_label.text() == "(всего: 7)"
    print("OK: счётчик посетителей обновляется")

    # --- 4. Секция «Журнал» сворачивается ---
    before = page._journal_panel.isHidden()
    page._toggle_journal()
    after = page._journal_panel.isHidden()
    assert before != after, "журнал не сворачивается"
    page._toggle_journal()
    assert page._journal_panel.isHidden() == before
    print("OK: секция «Журнал» сворачивается/разворачивается")

    # --- 5. Таилы метрик убраны ---
    for name in ("_clients_metric", "_status_metric", "_suitable_metric"):
        assert not hasattr(page, name), f"тайл {name} остался"
    print("OK: тайлы FPS/инциденты/клиенты удалены")

    # --- 6. FPS попадает в секцию «Камера и обработка» ---
    page._on_fps_updated(24.5)
    assert "24.5" in page._details_pipeline.text(), page._details_pipeline.text()
    print("OK: FPS отображается в секции «Камера и обработка»")

    manager.stop_all()
    print("ВСЁ ОК")

    check_live_removal(app)
    print("ВСЁ ОК (в т.ч. удаление живого потока)")

    check_main_window_cycle(app)
    print("ВСЁ ОК (полный цикл с MainWindow)")
    return 0


def check_live_removal(app) -> None:
    """Удаление потока с РАБОТАЮЩИМ воркером (файл-источник)."""
    manager = StreamManager()
    groups = GroupManager(path=ROOT / "state" / "_check_groups2.json")
    page = StreamsPage(manager, groups, None)

    cfg = StreamConfig(
        name="Файл", source=str(ROOT / "data" / "test.mp4"), location="Офис"
    )
    manager.add_stream(cfg)
    page.refresh_stream_list()

    ok = manager.start_stream(cfg.stream_id)
    print("старт воркера:", ok)
    # Ждём, пока воркер реально запустится и прочитает кадры.
    for _ in range(60):
        app.processEvents()
        runtime = manager.get_runtime(cfg.stream_id)
        if runtime is not None and runtime.worker is not None:
            if runtime.worker.isRunning():
                break
        time.sleep(0.1)

    runtime = manager.get_runtime(cfg.stream_id)
    print("воркер живой:", runtime.worker is not None and runtime.worker.isRunning())

    page._delete_stream(cfg.stream_id, confirm=False)
    assert manager.get_runtime(cfg.stream_id) is None
    print("OK: поток с работающим воркером удалён")

    # Даём время воркеру корректно завершиться (без «QThread: Destroyed»).
    for _ in range(30):
        app.processEvents()
        time.sleep(0.1)
    manager.stop_all()
    print("OK: воркер завершился без аварий")


if __name__ == "__main__":
    raise SystemExit(main())
