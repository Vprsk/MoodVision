"""Точка входа десктопного приложения «EmoSense».

Запуск:  python app/main.py
(из корня проекта, чтобы корректно разрешались импорты contracts/models).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Гарантируем, что корень проекта в sys.path (для запуска из любой директории).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Настраиваем логирование как можно раньше — до создания QApplication и
# импортов UI-модулей, чтобы ключевые ошибки и события попали в терминал.
from app.core.logging import setup_logging  # noqa: E402

setup_logging()

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.ui.style_qss import QSS


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("EmoSense")
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
