"""Персистентность: сохранение/загрузка списка видеопотоков.

Потоки сохраняются в JSON-файл (по умолчанию — streams.json в корне
проекта) при закрытии приложения и восстанавливаются при старте. Файл
содержит RTSP/HTTP-адреса, которые могут включать учётные данные, поэтому
он добавлен в .gitignore и не должен коммититься.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List

from app.core.logging import get_logger
from app.core.stream_config import StreamConfig

logger = get_logger("persistence")

# Корень проекта = два уровня выше этого файла (app/core/persistence.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Папка для runtime-состояния приложения (streams.json, groups.json и т.п.),
# чтобы не захламлять корень проекта сохранениями.
STATE_DIR = PROJECT_ROOT / "state"
DEFAULT_STORE_PATH = STATE_DIR / "streams.json"


def _resolve_path(path: str | os.PathLike | None) -> Path:
    return Path(path) if path is not None else DEFAULT_STORE_PATH


def write_json_atomic(path: str | os.PathLike, payload: dict) -> Path:
    """Атомарно записывает payload как JSON в указанный путь.

    Запись идёт во временный файл в той же директории, затем атомарно
    заменяет целевой через os.replace — защита от повреждённого файла,
    если приложение закроют во время сохранения. Возвращает путь к файлу.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.stem}-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, target)
    finally:
        # Если replace не удался — уберём временный файл.
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return target


def save_streams(
    configs: List[StreamConfig], path: str | os.PathLike | None = None
) -> Path:
    """Сохраняет список конфигов потоков в JSON (атомарно).

    Атомарная запись через временный файл + os.replace защищает от
    повреждённого файла, если приложение закроют во время сохранения.
    """
    target = _resolve_path(path)
    payload = {"version": 1, "streams": [c.to_dict() for c in configs]}
    try:
        write_json_atomic(target, payload)
    except OSError as exc:  # noqa: BLE001
        logger.error("Не удалось сохранить потоки в %s: %s", target, exc)
        raise
    logger.debug("Сохранено %d потоков в %s", len(configs), target)
    return target


def load_streams(path: str | os.PathLike | None = None) -> List[StreamConfig]:
    """Загружает конфиги потоков из JSON. При отсутствии/ошибке — []."""
    target = _resolve_path(path)
    if not target.exists():
        logger.debug("Файл потоков не найден: %s", target)
        return []
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
        logger.error("Не удалось загрузить потоки из %s: %s", target, exc)
        return []

    raw_streams = data.get("streams", []) if isinstance(data, dict) else []
    configs: List[StreamConfig] = []
    for item in raw_streams:
        try:
            config = StreamConfig.from_dict(item)
        except Exception:  # noqa: BLE001 — пропускаем битые записи
            continue
        # Минимальная валидация: без источника поток бесполезен.
        if config.source.strip():
            configs.append(config)
    return configs
