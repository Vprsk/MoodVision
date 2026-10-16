"""Настройка логирования приложения «EmoSense».

Ключевые события и ошибки, которые уже показываются в UI (статусы потоков,
ошибки модели, переподключения RTSP, сбои сохранения/загрузки и т.п.),
дублируются в терминал (stderr) через стандартный модуль ``logging``.

Уровень логирования можно задать переменной окружения ``EMOSENSE_LOG_LEVEL``
(по умолчанию ``INFO``). Допустимые значения: DEBUG, INFO, WARNING, ERROR.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Сохраняем приложение шумных сторонних библиотек на одном уровне, чтобы
# они не заваливали терминал отладочным шумом.
_THIRD_PARTY_LEVEL = logging.WARNING

_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int | str | None = None) -> logging.Logger:
    """Настраивает корневой логгер с выводом в консоль (stderr).

    Вызывать один раз в точке входа (``app/main.py``) до создания
    ``QApplication``. Если уровень не передан, берется из
    ``EMOSENSE_LOG_LEVEL`` или ``INFO``.
    """
    if level is None:
        level = os.environ.get("EMOSENSE_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Очищаем существующие хендлеры, чтобы при повторном вызове не дублировать
    # вывод (например, при перезапуске в одном процессе).
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console_handler)

    # Подавляем слишком многословное логирование сторонних библиотек.
    for name in ("cv2", "torch", "transformers", "ultralytics", "PIL"):
        logging.getLogger(name).setLevel(_THIRD_PARTY_LEVEL)

    logger = logging.getLogger("emose")
    logger.debug("Логирование инициализировано (уровень=%s)", logging.getLevelName(level))
    return logger


def get_logger(name: str) -> logging.Logger:
    """Возвращает именованный логгер в пространстве ``emose``."""
    return logging.getLogger(f"emose.{name}")
