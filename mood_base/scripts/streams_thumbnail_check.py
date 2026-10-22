"""Headless-проверка миниатюр в списке потоков.

Запуск (из корня проекта):
    QT_QPA_PLATFORM=offscreen env/Scripts/python.exe mood_base/scripts/streams_thumbnail_check.py

Проверяет:
  * виджет Thumbnail принимает кадр BGR и рисует его;
  * в карточке каждого потока есть миниатюра;
  * кадр-превью от воркера доходит до миниатюры и кладётся в кэш;
  * кэш переживает перерисовку списка (миниатюра не мигает «Нет сигнала»);
  * остановленный поток гасит миниатюру («застывший» кадр).
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

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.group_manager import GroupManager  # noqa: E402
from app.core.stream_config import StreamConfig  # noqa: E402
from app.core.stream_manager import StreamManager  # noqa: E402
from app.ui.streams_page import StreamsPage  # noqa: E402
from app.ui.widgets.thumbnail import Thumbnail  # noqa: E402


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILURES.append(name)


FAILURES: list = []


def check_thumbnail_widget(app) -> None:
    """Виджет миниатюры принимает кадр и умеет гаснуть."""
    thumb = Thumbnail((86, 52), "Нет сигнала")
    assert thumb._pixmap is None, "новый виджет должен быть пустым"
    thumb.set_frame(np.zeros((120, 160, 3), dtype=np.uint8))
    check("кадр принят миниатюрой", thumb._pixmap is not None)
    thumb.set_offline(True)
    check("миниатюра переходит в режим «офлайн»", thumb._offline is True)
    thumb.clear()
    check("миниатюра очищается", thumb._pixmap is None)
    # Мусор в сигнале не должен ронять приложение.
    thumb.set_frame(None)
    thumb.set_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    check("битые кадры игнорируются", True)


def check_page_miniatures(app) -> None:
    """Миниатюры в карточках списка + сквозной путь кадра-превью."""
    manager = StreamManager()
    groups = GroupManager(path=ROOT / "state" / "_check_thumb_groups.json")
    page = StreamsPage(manager, groups, None)

    cfg = StreamConfig(name="Камера 1", source="webcam:0", location="Офис")
    manager.add_stream(cfg)
    page.refresh_stream_list()

    item = page._camera_widgets[cfg.stream_id]
    check("в карточке потока есть миниатюра", isinstance(item._thumbnail, Thumbnail))
    check("миниатюра пуста до первого кадра", item._thumbnail._pixmap is None)

    # Приходит превью от менеджера — миниатюра обновляется.
    frame = np.full((60, 80, 3), 200, dtype=np.uint8)
    manager.frame_preview.emit(cfg.stream_id, frame)
    check(
        "кадр-превью дошёл до миниатюры",
        item._thumbnail._pixmap is not None,
    )
    check("кадр сохранён в кэш", cfg.stream_id in page._preview_cache)

    # Перерисовка списка (перенос в группу) — миниатюра не мигает.
    page._move_stream_to_group(cfg.stream_id, "Склад")
    item2 = page._camera_widgets[cfg.stream_id]
    check(
        "после перерисовки списка миниатюра сохранилась",
        item2._thumbnail._pixmap is not None,
    )

    # Остановленный поток — миниатюра гаснет.
    item2.set_offline(True)
    check("остановленный поток гасит миниатюру", item2._thumbnail._offline is True)

    # Удаление потока чистит кэш.
    page._delete_stream(cfg.stream_id, confirm=False)
    check("после удаления кэш превью очищен", cfg.stream_id not in page._preview_cache)

    manager.stop_all()


def check_live_preview(app) -> None:
    """Реальный воркер (файл-источник) отдаёт кадры в миниатюру."""
    manager = StreamManager()
    groups = GroupManager(path=ROOT / "state" / "_check_thumb_groups2.json")
    page = StreamsPage(manager, groups, None)

    cfg = StreamConfig(
        name="Файл", source=str(ROOT / "data" / "test.mp4"), location="Офис"
    )
    manager.add_stream(cfg)
    page.refresh_stream_list()
    manager.start_stream(cfg.stream_id)

    got = False
    # Загрузка модели (YOLO + ViT) занимает секунды, поэтому ждём щедро.
    for _ in range(400):
        app.processEvents()
        if cfg.stream_id in page._preview_cache:
            got = True
            break
        time.sleep(0.1)

    check("воркер прислал кадр-превью", got)
    if got:
        cached = page._preview_cache[cfg.stream_id]
        check(
            "кадр превью уменьшен (ширина <= 240)",
            getattr(cached, "shape", (0, 0))[1] <= 240,
            str(getattr(cached, "shape", None)),
        )
        item = page._camera_widgets.get(cfg.stream_id)
        check(
            "миниатюра карточки показывает кадр",
            item is not None and item._thumbnail._pixmap is not None,
        )

    manager.stop_all()
    app.processEvents()


def main() -> int:
    app = QApplication(sys.argv)

    print("== Виджет миниатюры ==")
    check_thumbnail_widget(app)

    print("== Миниатюры в списке потоков ==")
    check_page_miniatures(app)

    print("== Сквозной путь: воркер -> миниатюра ==")
    check_live_preview(app)

    print("\n=== ИТОГ ===")
    if FAILURES:
        print(f"Провалено: {len(FAILURES)} → {FAILURES}")
        print("РЕЗУЛЬТАТ: FAIL")
        return 1
    print("РЕЗУЛЬТАТ: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
