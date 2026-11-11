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
        from mood_base.models import CurrentYoloVitModel

        return CurrentYoloVitModel(**params)

    def _scene1_factory(**params: Any) -> BaseFaceEmotionModel:
        from mood_base.models import Scene1FaceAudioModel

        return Scene1FaceAudioModel(**params)

    def _scene2_factory(**params: Any) -> BaseFaceEmotionModel:
        from mood_base.models import Scene2FaceBodyAudioModel

        return Scene2FaceBodyAudioModel(**params)

    def _scene3_factory(**params: Any) -> BaseFaceEmotionModel:
        from mood_base.models import Scene3BodyAudioModel

        return Scene3BodyAudioModel(**params)

    def _scene4_factory(**params: Any) -> BaseFaceEmotionModel:
        from mood_base.models import Scene4FaceAudioEfficientNetB0Model

        return Scene4FaceAudioEfficientNetB0Model(**params)

    # 0. Базовый пайплайн (оставляем первым для обратной совместимости и тестов)
    register_pipeline(
        PipelineInfo(
            key="yolo_vit",
            label="YOLOv8-face + ViT (текущий)",
            factory=_yolo_vit_factory,
            defaults={
                "model_path": "mood_base/model/yolov8n-face.pt",
                "tracker_config": "configs/bytetrack.yaml",
                "conf_threshold": 0.45,
                "emotion_conf_threshold": 0.35,
                "yolo_imgsz": 416,
                "analyze_every_n_frames": 5,
                "frame_scale": 0.5,
                "batch_size": 8,
            },
            description=(
                "Детекция и трекинг лиц (YOLOv8-face + ByteTrack) с "
                "классификацией 7 базовых эмоций (ViT)."
            ),
        )
    )

    # 1. Сцена 1: Лицо + Звук (прикассовая зона / крупный план)
    register_pipeline(
        PipelineInfo(
            key="scene1_face_audio",
            label="Сцена 1: Лицо + Звук",
            factory=_scene1_factory,
            defaults={
                "model_path": "mood_base/model/yolov8n-face.pt",
                "tracker_config": "configs/bytetrack.yaml",
                "conf_threshold": 0.45,
                "emotion_conf_threshold": 0.35,
                "min_face_px": 48,
                "alert_threshold": 0.65,
                "temporal_window": 15,
                "yolo_imgsz": 416,
                "analyze_every_n_frames": 5,
                "frame_scale": 0.5,
            },
            description=(
                "Крупный план (стойка/касса): трекинг лиц YOLOv8-face + "
                "фильтр качества лица (Quality Gate) + мимика (ViT) + эмоции по голосу."
            ),
        )
    )

    # 2. Сцена 2: Лицо + Звук + Тело (средняя дистанция / очередь / зал)
    register_pipeline(
        PipelineInfo(
            key="scene2_face_body_audio",
            label="Сцена 2: Лицо + Звук + Тело (Каскад)",
            factory=_scene2_factory,
            defaults={
                "face_model_path": "mood_base/model/yolov8n-face.pt",
                "pose_model_path": "mood_base/model/yolov8n-pose.pt",
                "tracker_config": "configs/bytetrack.yaml",
                "conf_threshold": 0.45,
                "body_conf_threshold": 0.40,
                "emotion_conf_threshold": 0.35,
                "min_face_px": 64,
                "alert_threshold": 0.65,
                "temporal_window": 15,
                "yolo_imgsz": 416,
                "analyze_every_n_frames": 5,
                "frame_scale": 0.5,
            },
            description=(
                "Средняя дистанция (очередь/зал): непрерывный трекинг по телу "
                "(YOLOv8-pose, 3 уровня языка тела) + условная мимика при видимом "
                "лице (≥ 64 px) + эмоции по голосу."
            ),
        )
    )

    # 3. Сцена 3: Звук + Тело (общий план / дальняя дистанция без детекции лиц)
    register_pipeline(
        PipelineInfo(
            key="scene3_body_audio",
            label="Сцена 3: Звук + Тело (Общий план)",
            factory=_scene3_factory,
            defaults={
                "pose_model_path": "mood_base/model/yolov8n-pose.pt",
                "tracker_config": "configs/bytetrack.yaml",
                "body_conf_threshold": 0.40,
                "emotion_conf_threshold": 0.35,
                "alert_threshold": 0.65,
                "temporal_window": 15,
                "yolo_imgsz": 416,
                "analyze_every_n_frames": 5,
                "frame_scale": 0.5,
            },
            description=(
                "Дальний/общий план (лица < 40 px): модели лица отключены, "
                "эмоции определяются по 3 уровням языка тела (YOLOv8-pose + позы + ViT) "
                "и ведущему аудиоканалу."
            ),
        )
    )

    # 4. Сцена 4: Лицо + Звук (EfficientNet-B0 вместо ViT)
    register_pipeline(
        PipelineInfo(
            key="scene4_face_audio_efficientnet_b0",
            label="Сцена 4: Лицо + Звук (EfficientNet-B0)",
            factory=_scene4_factory,
            defaults={
                "model_path": "mood_base/model/yolov8n-face.pt",
                "tracker_config": "configs/bytetrack.yaml",
                "conf_threshold": 0.45,
                "emotion_conf_threshold": 0.35,
                "min_face_px": 48,
                "alert_threshold": 0.65,
                "temporal_window": 15,
                "yolo_imgsz": 416,
                "analyze_every_n_frames": 5,
                "frame_scale": 0.5,
                "emotion_model_name": "enet_b0_8_best_afew",
            },
            description=(
                "Крупный план (стойка/касса): трекинг лиц YOLOv8-face + "
                "фильтр качества лица (Quality Gate) + мимика EfficientNet-B0 "
                "(hsemotion enet_b0_8_best_afew) + эмоции по голосу."
            ),
        )
    )


_register_builtin_pipelines()