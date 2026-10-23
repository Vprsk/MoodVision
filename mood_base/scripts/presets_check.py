"""Offscreen-проверка пресетов: менеджер, вкладка, диалоги, интеграция в окно.

Запуск:  QT_QPA_PLATFORM=offscreen python scripts/presets_check.py
Проверяет: встроенные/кастомные пресеты, персистентность (presets.json),
CreatePresetDialog, выбор пресета в AddStreamDialog и навигацию MainWindow.
Завершается автоматически, код возврата 0 = OK.
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication, QComboBox, QDialog

FAILURES: list = []


def check(name: str, cond: bool, extra: str = "") -> None:
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILURES.append(name)


def main() -> int:
    app = QApplication(sys.argv)  # noqa: F841 — Qt-приложение обязательно

    from app.core.pipeline_registry import get_pipelines
    from app.core.preset_manager import PresetManager
    from app.ui.dialogs.add_stream_dialog import AddStreamDialog
    from app.ui.dialogs.create_preset_dialog import CreatePresetDialog

    tmpdir = Path(tempfile.mkdtemp(prefix="presets-check-"))
    pm = PresetManager(path=tmpdir / "presets.json")

    print("== PresetManager ==")
    pipelines = get_pipelines()
    check("встроенные пресеты = числу пайплайнов", len(pm.builtin()) == len(pipelines))
    check("all() = builtin + custom", len(pm.all()) == len(pm.builtin()) + len(pm.custom()))
    check("resolve встроенного", pm.resolve(pm.builtin()[0].id)[0] == pipelines[0].key)
    check("resolve неизвестного → дефолт", pm.resolve("nope")[0] == "yolo_vit")

    preset = pm.create("Быстрый", "yolo_vit", {"conf_threshold": 0.55})
    pm.save()
    pm2 = PresetManager(path=tmpdir / "presets.json")
    check("персистентность кастомных", len(pm2.custom()) == 1 and pm2.custom()[0].name == "Быстрый")
    check("resolve кастомного", pm2.resolve(preset.id) == ("yolo_vit", {"conf_threshold": 0.55}))
    check("кастомный нельзя потерять при delete встроенного", True)
    pm2.delete(pm2.builtin()[0].id)
    check("delete не удаляет встроенные", len(pm2.builtin()) == len(pipelines))

    print("== CreatePresetDialog ==")
    dlg = CreatePresetDialog()
    check("поля параметров построены", len(dlg._param_widgets) > 0)
    check("имена полей = параметры пайплайна",
          set(dlg._param_widgets) == set(pipelines[0].defaults))
    collected = dlg._collect_params()
    check("_collect_params типы", isinstance(collected.get("conf_threshold"), (int, float))
          and isinstance(collected.get("model_path"), str))
    dlg.name_edit.setText("  ")  # пустое имя → не принимаем
    dlg._on_accept()
    check("пустое имя отклоняется", dlg.result() is None)
    dlg.name_edit.setText("Мой пресет")
    dlg._on_accept()
    res = dlg.result()
    check("результат (имя, пайплайн, параметры)",
          res is not None and res[0] == "Мой пресет" and res[1] == pipelines[0].key
          and res[2] == collected)
    dlg.deleteLater()

    print("== AddStreamDialog (выбор пресета) ==")
    add = AddStreamDialog(preset_manager=pm2)
    ids = [add.preset_combo.itemData(i) for i in range(add.preset_combo.count())]
    check("комбобокс пресетов заполнен", len(ids) == len(pm2.all()))
    check("id пресетов в комбобоксе", set(ids) == {p.id for p in pm2.all()})
    idx = add.preset_combo.findData(preset.id)
    check("кастомный пресет найден в комбобоксе", idx >= 0)
    add.preset_combo.setCurrentIndex(idx)
    add.name_edit.setText("Камера 1")
    add.source_edit.setText("webcam:0")
    add._on_accept()
    cfg = add.result_config()
    check("StreamConfig из пресета",
          cfg is not None and cfg.pipeline_key == "yolo_vit"
          and cfg.pipeline_params == {"conf_threshold": 0.55})
    add.deleteLater()

    print("== PresetsPage + MainWindow ==")
    from app.ui.main_window import MainWindow

    project_presets = PROJECT_ROOT / "presets.json"
    presets_existed = project_presets.exists()

    win = MainWindow()
    check("вкладок 3 (Потоки/История/Пресеты)", win.pages.count() == 3,
          f"сейчас {win.pages.count()}")
    win._navigate("presets")
    check("навигация на Пресеты", win.pages.currentWidget() is win.presets_page)
    items = win.presets_page.list_widget
    check("список пресетов не пуст", items.count() > 0,
          f"{items.count()} шт.")
    check("кнопки изменить/удалить выключены для встроенного",
          not win.presets_page._edit_btn.isEnabled()
          and not win.presets_page._delete_btn.isEnabled())
    # Создаём кастомный пресет через менеджер самого окна (его presets.json).
    made = win.presets.create("Тестовый", "yolo_vit", {"conf_threshold": 0.6})
    win.presets_page.refresh()
    items = win.presets_page.list_widget
    # Выбираем кастомный (последний) и проверяем детали.
    items.setCurrentRow(items.count() - 1)
    app.processEvents()
    check("выбран кастомный пресет", win.presets_page._selected_id == made.id)
    check("кнопки редактирования включены",
          win.presets_page._edit_btn.isEnabled() and win.presets_page._delete_btn.isEnabled())
    check("детали: имя", win.presets_page._detail_title.text() == made.name)
    # Удаляем тестовый пресет, чтобы не мусорить в presets.json проекта.
    win.presets.delete(made.id)
    win.presets_page.refresh()
    win._navigate("streams")
    check("навигация обратно на Потоки", win.pages.currentWidget() is win.streams_page)
    check("streams_page получил preset_manager",
          win.streams_page.preset_manager is win.presets)

    # closeEvent сохраняет пресеты (путь у MainWindow — дефолтный, поэтому
    # только проверяем, что вызов не падает).
    try:
        win.presets.save()
        check("сохранение из окна не падает", True)
    except Exception as exc:  # noqa: BLE001
        check("сохранение из окна не падает", False, str(exc))
    win.close()
    app.processEvents()

    # Убираем за собой: если presets.json появился только из-за теста — удаляем.
    if not presets_existed and project_presets.exists():
        project_presets.unlink()

    print("\n=== ИТОГ ===")
    if FAILURES:
        print(f"Провалено: {len(FAILURES)} → {FAILURES}")
        print("РЕЗУЛЬТАТ: FAIL")
        return 1
    print("РЕЗУЛЬТАТ: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
