"""Управление группами видеопотоков (группировка в сайдбаре «Потоки»).

Группа — это именованный контейнер для потоков. Принадлежность потока к
группе определяется полем StreamConfig.location. Менеджер хранит упорядоченный
список групп, их состояние «свёрнуто/развёрнуто» и персистит это в JSON,
чтобы пустые группы и порядок сохранялись между запусками.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.core.logging import get_logger
from app.core.persistence import STATE_DIR, write_json_atomic

logger = get_logger("group_manager")

DEFAULT_GROUPS_PATH = STATE_DIR / "groups.json"
DEFAULT_GROUP = "Без группы"


@dataclass
class Group:
    """Одна группа потоков."""

    name: str
    order: int = 0
    collapsed: bool = False


class GroupManager:
    """Хранит и управляет списком групп. Не зависит от Qt."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_GROUPS_PATH
        self._groups: Dict[str, Group] = {}
        self.load()

    # --- Персистентность ----------------------------------------------------

    def load(self) -> None:
        """Загружает группы из JSON. При отсутствии/ошибке — пустой список."""
        self._groups = {}
        if not self._path.exists():
            logger.debug("Файл групп не найден: %s", self._path)
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
            logger.error("Не удалось загрузить группы из %s: %s", self._path, exc)
            return
        for item in data.get("groups", []):
            try:
                name = str(item["name"]).strip()
            except (KeyError, TypeError):
                continue
            if name:
                self._groups[name] = Group(
                    name=name,
                    order=int(item.get("order", 0)),
                    collapsed=bool(item.get("collapsed", False)),
                )

    def save(self) -> None:
        """Сохраняет группы в JSON (атомарно через временный файл)."""
        payload = {"groups": [g.__dict__ for g in self.ordered()]}
        write_json_atomic(self._path, payload)
        logger.debug("Сохранено %d групп в %s", len(payload["groups"]), self._path)

    # --- CRUD групп ---------------------------------------------------------

    def all(self) -> List[Group]:
        """Возвращает группы, отсортированные по order."""
        return self.ordered()

    def ordered(self) -> List[Group]:
        return sorted(self._groups.values(), key=lambda g: g.order)

    def exists(self, name: str) -> bool:
        return name in self._groups

    def ensure(self, name: str) -> Group:
        """Возвращает группу, создавая её при необходимости.

        Нужно, чтобы группы, впервые увиденные у загруженных потоков,
        автоматически попадали в список (и сохранялись).
        """
        name = (name or "").strip() or DEFAULT_GROUP
        if name not in self._groups:
            order = max((g.order for g in self._groups.values()), default=-1) + 1
            self._groups[name] = Group(name=name, order=order)
        return self._groups[name]

    def create(self, name: str) -> Optional[Group]:
        """Создаёт новую группу. Возвращает None, если имя занято/пусто."""
        name = (name or "").strip()
        if not name or name in self._groups:
            return None
        order = max((g.order for g in self._groups.values()), default=-1) + 1
        group = Group(name=name, order=order)
        self._groups[name] = group
        return group

    def rename(self, old: str, new: str) -> bool:
        """Переименовывает группу. False, если old нет или new занято/пусто."""
        new = (new or "").strip()
        if old not in self._groups or not new or new == old:
            return False
        if new in self._groups:
            return False
        group = self._groups.pop(old)
        group.name = new
        self._groups[new] = group
        return True

    def delete(self, name: str) -> None:
        """Удаляет группу (потоки в ней должны быть перенесены вызывающим)."""
        self._groups.pop(name, None)

    # --- Состояние свёрнутости ---------------------------------------------

    def is_collapsed(self, name: str) -> bool:
        group = self._groups.get(name)
        return group.collapsed if group else False

    def set_collapsed(self, name: str, collapsed: bool) -> None:
        group = self._groups.get(name)
        if group is not None:
            group.collapsed = collapsed

    def toggle(self, name: str) -> bool:
        """Переключает свёрнутость группы, возвращает новое состояние."""
        group = self._groups.get(name)
        if group is None:
            return False
        group.collapsed = not group.collapsed
        return group.collapsed
