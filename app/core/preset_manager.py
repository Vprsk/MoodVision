"""Управление пресетами обработки видеопотоков.

Пресет — это именованный набор «пайплайн + параметры», который выбирается
при добавлении потока. Так пользователю не нужно каждый раз вручную
настраивать пороги/частоту — достаточно выбрать готовый пресет.

Два типа пресетов:
  * встроенные (built-in) — по одному на каждый зарегистрированный пайплайн,
    с параметрами по умолчанию из реестра; не редактируются и не удаляются;
  * кастомные (custom) — создаются пользователем во вкладке «Пресеты»
    (выбор пайплайна + свои параметры), редактируются, удаляются и
    сохраняются в presets.json между запусками.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.persistence import PROJECT_ROOT

DEFAULT_PRESETS_PATH = PROJECT_ROOT / "presets.json"
BUILTIN_PREFIX = "builtin:"


@dataclass
class Preset:
    """Один пресет обработки."""

    id: str
    name: str
    pipeline_key: str
    params: Dict[str, Any] = field(default_factory=dict)
    is_custom: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "pipeline_key": self.pipeline_key,
            "params": dict(self.params or {}),
            "is_custom": self.is_custom,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Preset":
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "Пресет")),
            pipeline_key=str(data.get("pipeline_key", "yolo_vit")),
            params=dict(data.get("params") or {}),
            is_custom=bool(data.get("is_custom", True)),
            description=str(data.get("description", "")),
        )


class PresetManager:
    """Хранит встроенные и кастомные пресеты, персистит кастомные в JSON."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_PRESETS_PATH
        self._custom: List[Preset] = []
        self.load()

    # --- Встроенные пресеты (из реестра пайплайнов) ------------------------

    def builtin(self) -> List[Preset]:
        """Возвращает встроенные пресеты по одному на каждый пайплайн."""
        from app.core.pipeline_registry import get_pipelines

        result: List[Preset] = []
        for info in get_pipelines():
            result.append(
                Preset(
                    id=f"{BUILTIN_PREFIX}{info.key}",
                    name=info.label,
                    pipeline_key=info.key,
                    params=dict(info.defaults),
                    is_custom=False,
                    description=info.description,
                )
            )
        return result

    def all(self) -> List[Preset]:
        """Все пресеты: сначала встроенные, затем кастомные."""
        return self.builtin() + list(self._custom)

    def custom(self) -> List[Preset]:
        """Только кастомные пресеты."""
        return list(self._custom)

    def get(self, preset_id: str) -> Optional[Preset]:
        """Находит пресет по id (среди всех)."""
        for preset in self.all():
            if preset.id == preset_id:
                return preset
        return None

    def default_preset(self) -> Preset:
        """Пресет по умолчанию (первый встроенный, иначе первый кастомный)."""
        builtin = self.builtin()
        if builtin:
            return builtin[0]
        if self._custom:
            return self._custom[0]
        return Preset(
            id=f"{BUILTIN_PREFIX}yolo_vit", name="По умолчанию", pipeline_key="yolo_vit"
        )

    def resolve(self, preset_id: str) -> Tuple[str, Dict[str, Any]]:
        """Возвращает (pipeline_key, params) для пресета.

        Если пресет не найден — откат к пресету по умолчанию.
        """
        preset = self.get(preset_id) or self.default_preset()
        return preset.pipeline_key, dict(preset.params or {})

    # --- CRUD кастомных пресетов -------------------------------------------

    def create(self, name: str, pipeline_key: str, params: Dict[str, Any]) -> Preset:
        """Создаёт кастомный пресет и возвращает его."""
        preset = Preset(
            id=uuid.uuid4().hex[:8],
            name=(name or "").strip() or "Пресет",
            pipeline_key=pipeline_key,
            params=dict(params or {}),
            is_custom=True,
        )
        self._custom.append(preset)
        return preset

    def update(
        self,
        preset_id: str,
        name: Optional[str] = None,
        pipeline_key: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Обновляет кастомный пресет. False, если не найден или не кастомный."""
        for preset in self._custom:
            if preset.id == preset_id:
                if name is not None and name.strip():
                    preset.name = name.strip()
                if pipeline_key is not None:
                    preset.pipeline_key = pipeline_key
                if params is not None:
                    preset.params = dict(params)
                return True
        return False

    def delete(self, preset_id: str) -> None:
        """Удаляет кастомный пресет (встроенные не затрагиваются)."""
        self._custom = [p for p in self._custom if p.id != preset_id]

    # --- Персистентность ---------------------------------------------------

    def save(self) -> None:
        """Сохраняет кастомные пресеты в JSON (атомарно через временный файл)."""
        payload = {"presets": [p.to_dict() for p in self._custom]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".presets-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def load(self) -> None:
        """Загружает кастомные пресеты из JSON. При ошибке — пустой список."""
        self._custom = []
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        for item in data.get("presets", []):
            try:
                preset = Preset.from_dict(item)
            except Exception:  # noqa: BLE001 — пропускаем битые записи
                continue
            if preset.id and preset.is_custom:
                self._custom.append(preset)

