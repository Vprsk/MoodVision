"""Настройка логирования приложения «EmoSense».

Ключевые события и ошибки, которые уже показываются в UI (статусы потоков,
ошибки модели, переподключения RTSP, сбои сохранения/загрузки и т.п.),
дублируются в терминал (stderr) через стандартный модуль ``logging``.

Дополнительно всё пишется в файл ``logs/emose.log`` (с ротацией: до 2 МБ
на файл, хранится 3 старых копии). Файловый хендлер работает на уровне
DEBUG — он подробнее консольного, чтобы при разборе инцидентов (в т.ч. при
запуске приложения без терминала, когда консоль недоступна) была полная
картина. Уровень консоли по умолчанию INFO.

Уровень логирования консоли можно задать переменной окружения
``EMOSENSE_LOG_LEVEL`` (по умолчанию ``INFO``). Допустимые значения:
DEBUG, INFO, WARNING, ERROR.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Сохраняем приложение шумных сторонних библиотек на одном уровне, чтобы
# они не заваливали терминал отладочным шумом.
_THIRD_PARTY_LEVEL = logging.WARNING

_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# Корень проекта: <root>/app/core/logging.py -> parents[2] = <root>.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Папка файловых логов (создаётся при необходимости).
LOGS_DIR = _PROJECT_ROOT / "logs"
# Имя основного файла логов внутри LOGS_DIR.
LOG_FILENAME = "emose.log"
# Ротация файловых логов: не разрастается без ограничения.
_MAX_LOG_BYTES = 2 * 1024 * 1024  # 2 МБ на файл
_LOG_BACKUPS = 3  # emose.log.1, .2, .3 — старые копии


def setup_logging(level: int | str | None = None) -> logging.Logger:
    """Настраивает корневой логгер: консоль (stderr) + файл с ротацией.

    Вызывать один раз в точке входа (``app/main.py``) до создания
    ``QApplication``. Если уровень не передан, берется из
    ``EMOSENSE_LOG_LEVEL`` или ``INFO`` (уровень консоли; файл всегда
    пишется на DEBUG).
    """
    if level is None:
        level = os.environ.get("EMOSENSE_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    # Корень пропускает DEBUG — подробности отфильтруют хендлеры ниже
    # (консоль — по уровню из env, файл — всегда DEBUG).
    root.setLevel(logging.DEBUG)

    # Очищаем существующие хендлеры, чтобы при повторном вызове не дублировать
    # вывод (например, при перезапуске в одном процессе).
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console_handler)

    # Файловый хендлер: пишет всё (DEBUG), даже если консоль недоступна.
    # Ошибка создания файла не должна ронять приложение.
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOGS_DIR / LOG_FILENAME,
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_LOG_BACKUPS,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        )
        root.addHandler(file_handler)
    except OSError as exc:  # noqa: BLE001 — не роняем запуск из-за логов
        print(
            f"[logging] Не удалось открыть файл логов в {LOGS_DIR}: {exc}",
            file=sys.stderr,
        )

    # Подавляем слишком многословное логирование сторонних библиотек.
    for name in ("cv2", "torch", "transformers", "ultralytics", "PIL"):
        logging.getLogger(name).setLevel(_THIRD_PARTY_LEVEL)

    logger = logging.getLogger("emose")
    logger.debug("Логирование инициализировано (уровень=%s)", logging.getLevelName(level))
    return logger


def get_logger(name: str) -> logging.Logger:
    """Возвращает именованный логгер в пространстве ``emose``."""
    return logging.getLogger(f"emose.{name}")
