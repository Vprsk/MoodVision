"""Реестр пайплайнов обработки (Dependency Injection).

Позволяет выбирать и подставлять любую реализацию контракта
contracts.base.BaseFaceEmotionModel без изменения GUI-слоя.

Чтобы добавить новый пайплайн, достаточно реализовать контракт и
вызвать register_pipeline() — он автоматически появится в UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from contracts.base import BaseFaceEmotionModel

# Фабрика: принимает переопределяемые параметры и возвращает модель.
PipelineFactory = Callable[..., BaseFaceEmotionModel]


@dataclass
class PipelineInfo:
    """Метаданные пайплайна для UI и его создания."""

    key: str
    label: str
    factory: PipelineFactory
    defaults: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


_REGISTRY: Dict[str, PipelineInfo] = {}


def register_pipeline(info: PipelineInfo) -> None:
    """Регистрирует (или перезаписывает) пайплайн по его ключу."""
    _REGISTRY[info.key] = info


def get_pipelines() -> List[PipelineInfo]:
    """Возвращает список доступных пайплайнов (для выпадающих списков UI)."""
    return list(_REGISTRY.values())


def get_pipeline(key: str) -> Optional[PipelineInfo]:
    """Возвращает метаданные пайплайна по ключу или None."""
    return _REGISTRY.get(key)


def create_pipeline(key: str, **overrides: Any) -> BaseFaceEmotionModel:
    """Создаёт экземпляр модели выбранного пайплайна.

    Объединяет параметры по умолчанию пайплайна с переданными overrides,
    затем вызывает фабрику. Каждый вызов создаёт НОВЫЙ экземпляр модели —
    это важно, т.к. модели (YOLO+ByteTrack) содержат состояние треков и
    не предназначены для параллельного использования одним объектом.
    """
    info = _REGISTRY.get(key)
    if info is None:
        raise KeyError(
            f"Неизвестный пайплайн: {key!r}. Доступны: {list(_REGISTRY)}"
        )
    params: Dict[str, Any] = {**info.defaults, **overrides}
    return info.factory(**params)


def _register_builtin_pipelines() -> None:
    """Регистрирует встроенные пайплайны проекта.

    Импорт выполняется лениво внутри функции, чтобы тяжёлые зависимости
    (torch/ultralytics/transformers) загружались только при реальном
    создании модели, а не при импорте GUI.
    """

    def _yolo_vit_factory(**params: Any) -> BaseFaceEmotionModel:
        from models import CurrentYoloVitModel

        return CurrentYoloVitModel(**params)

    register_pipeline(
        PipelineInfo(
            key="yolo_vit",
            label="YOLOv8-face + ViT (текущий)",
            factory=_yolo_vit_factory,
            defaults={
                "model_path": "model/yolov8n-face.pt",
                "tracker_config": "configs/bytetrack.yaml",
                "conf_threshold": 0.45,
                "emotion_conf_threshold": 0.35,
                "yolo_imgsz": 416,
                "analyze_every_n_frames": 5,
                "frame_scale": 0.5,
            },
            description=(
                "Детекция и трекинг лиц (YOLOv8-face + ByteTrack) с "
                "классификацией 7 базовых эмоций (ViT)."
            ),
        )
    )


_register_builtin_pipelines()
