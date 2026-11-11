"""Классификатор эмоций на базе EfficientNet-B0 (модель enet_b0_8_best_afew из библиотеки hsemotion).

Реализует контракт contracts.base.BaseFaceEmotionModel.
Вход: BGR-кроп лица (numpy) от YOLOv8-face.
Выход: FaceDetection с эмоцией из схемы contracts.schemas.Emotion.

Классы модели enet_b0_8_best_afew (8 шт.):
    0: Anger, 1: Contempt, 2: Disgust, 3: Fear,
    4: Happiness, 5: Neutral, 6: Sadness, 7: Surprise

Сопоставление с нашим Emotion enum (7 шт. + UNKNOWN):
    Anger      -> ANGRY
    Contempt   -> ANGRY   (нет прямого аналога, ближайший негатив)
    Disgust    -> DISGUST
    Fear       -> FEAR
    Happiness  -> HAPPY
    Neutral    -> NEUTRAL
    Sadness    -> SAD
    Surprise   -> SURPRISE

Предобработка (строго как в hsemotion):
    Resize 224x224 -> ToTensor -> Normalize(ImageNet mean/std)

Веса скачиваются автоматически в ~/.hsemotion/ с GitHub при первом запуске.
"""

from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np
import torch

from app.core.logging import get_logger
from contracts.base import BaseFaceEmotionModel
from contracts.schemas import (
    BoundingBox,
    Emotion,
    FaceDetection,
    FrameResult,
)

logger = get_logger("efficientnet_b0")

# Имя модели в библиотеке hsemotion.
EFFICIENTNET_B0_MODEL_NAME = "enet_b0_8_best_afew"
# Входной размер EfficientNet-B0.
EFFICIENTNET_B0_IMG_SIZE = 224
# Порог уверенности по умолчанию (ниже -> UNKNOWN).
DEFAULT_EMOTION_CONF_THRESHOLD = 0.35

# Процессный кэш классификатора эмоций EfficientNet-B0.
_EFFICIENTNET_B0_CACHE: dict[tuple[str, int, str], Any] = {}
_EFFICIENTNET_B0_LOCK = threading.Lock()

# Маппинг меток модели hsemotion -> наш Emotion enum.
_HSEMOTION_TO_EMOTION: dict[str, Emotion] = {
    "Anger": Emotion.ANGRY,
    "Contempt": Emotion.ANGRY,
    "Disgust": Emotion.DISGUST,
    "Fear": Emotion.FEAR,
    "Happiness": Emotion.HAPPY,
    "Neutral": Emotion.NEUTRAL,
    "Sadness": Emotion.SAD,
    "Surprise": Emotion.SURPRISE,
}


def _softmax(x: np.ndarray) -> np.ndarray:
    """Стабильный softmax по последней оси."""
    x = x - np.max(x)
    e_x = np.exp(x)
    return e_x / e_x.sum()


def _download_model_path(model_name: str) -> str:
    """Скачивает чекпоинт модели hsemotion в локальный кэш и возвращает путь."""
    from hsemotion.facial_emotions import get_model_path
    return get_model_path(model_name)


def _load_efficientnet_b0_from_checkpoint(
    model_name: str, device: str
) -> tuple[Any, Any]:
    """Загружает веса EfficientNet-B0 из чекпоинта hsemotion.

    Чекпоинт enet_b0_8_best_afew был сохранён со старой версией timm, которая
    несовместима с timm ≥ 1.0 (разная структура DepthwiseSeparableConv).
    Вместо того чтобы патчить torch.load и добавлять недостающие атрибуты
    по одному, мы загружаем state_dict из чекпоинта и создаём новую модель
    EfficientNet-B0 через актуальный timm, затем загружаем веса в неё.

    Возвращает (model, classifier_weights, classifier_bias).
    """
    # timm ≥ 1.0 импортирует torch._six, удалённый в PyTorch 2.14+ / Python 3.14.
    # Добавляем заглушку, чтобы импорт timm прошёл успешно.
    import sys as _sys
    import types as _types
    if not hasattr(torch, "_six"):
        torch._six = _types.ModuleType("torch._six")  # type: ignore[attr-defined]
        import collections.abc as _collections_abc
        torch._six.container_abcs = _collections_abc  # type: ignore[attr-defined]
        _sys.modules["torch._six"] = torch._six  # type: ignore[attr-defined]

    import timm

    path = _download_model_path(model_name)

    # PyTorch ≥ 2.6 требует weights_only или allowlist. Чекпоинт из доверенного
    # источника (HSE-asavchenko на GitHub), поэтому используем weights_only=False.
    # Чекпоинт содержит ссылки на классы timm.layers.* (Conv2dSame, BatchNormAct2d,
    # DepthwiseSeparableConv и др.), которых может не быть в установленной версии
    # timm. Создаём модуль timm.layers с этими классами, чтобы pickle смог
    # десериализовать чекпоинт.
    import sys as _sys
    import types as _types

    # Убеждаемся, что timm.layers содержит все классы, которые могут
    # встречаться в чекпоинте (Conv2dSame, BatchNormAct2d,
    # DepthwiseSeparableConv и др.).
    try:
        import timm.layers as _tl
    except Exception:
        _tl = None

    _timm_layers_stub = None
    if _tl is None or "timm.layers" not in _sys.modules:
        _timm_layers_stub = _types.ModuleType("timm.layers")
        _sys.modules["timm.layers"] = _timm_layers_stub
        _tl = _timm_layers_stub

    # Добавляем недостающие классы в timm.layers.
    try:
        from timm.models._efficientnet_blocks import DepthwiseSeparableConv
        if not hasattr(_tl, "DepthwiseSeparableConv"):
            setattr(_tl, "DepthwiseSeparableConv", DepthwiseSeparableConv)
    except Exception:
        pass
    try:
        from timm.layers import Conv2dSame, BatchNormAct2d
        if not hasattr(_tl, "Conv2dSame"):
            setattr(_tl, "Conv2dSame", Conv2dSame)
        if not hasattr(_tl, "BatchNormAct2d"):
            setattr(_tl, "BatchNormAct2d", BatchNormAct2d)
    except Exception:
        pass

    _orig_torch_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        checkpoint = torch.load(path, map_location="cpu")
    finally:
        torch.load = _orig_torch_load
        if _timm_layers_stub is not None:
            _sys.modules.pop("timm.layers", None)

    # Извлекаем state_dict из объекта модели
    if hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Создаём модель EfficientNet-B0 через актуальный timm
    model = timm.create_model(
        "efficientnet_b0", pretrained=False, num_classes=8, drop_rate=0.0
    )
    model = model.to(device)
    model.eval()

    # Загружаем веса (strict=False, т.к. имена слоев могут немного отличаться)
    model.load_state_dict(state_dict, strict=False)

    # Извлекаем веса классификатора (как делает hsemotion)
    if isinstance(model.classifier, torch.nn.Sequential):
        classifier_weights = model.classifier[0].weight.cpu().data.numpy()
        classifier_bias = model.classifier[0].bias.cpu().data.numpy()
    else:
        classifier_weights = model.classifier.weight.cpu().data.numpy()
        classifier_bias = model.classifier.bias.cpu().data.numpy()

    # Заменяем классификатор на Identity, чтобы модель возвращала признаки
    model.classifier = torch.nn.Identity()

    return model, classifier_weights, classifier_bias


class _EfficientNetB0EmotionRecognizer:
    """Обёртка над моделью EfficientNet-B0 с интерфейсом, совместимым с
    hsemotion HSEmotionRecognizer (predict_emotions, idx_to_class)."""

    def __init__(
        self,
        model_name: str,
        device: str,
        is_mtl: bool = False,
    ) -> None:
        self.device = device
        self.is_mtl = is_mtl
        if "_7" in model_name:
            self.idx_to_class = {
                0: "Anger", 1: "Disgust", 2: "Fear", 3: "Happiness",
                4: "Neutral", 5: "Sadness", 6: "Surprise",
            }
        else:
            self.idx_to_class = {
                0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear",
                4: "Happiness", 5: "Neutral", 6: "Sadness", 7: "Surprise",
            }
        self.img_size = 224 if "_b0_" in model_name else 260

        self.model, self.classifier_weights, self.classifier_bias = (
            _load_efficientnet_b0_from_checkpoint(model_name, device)
        )

    def extract_features(self, face_img: Any) -> Any:
        import numpy as np
        from PIL import Image
        from torchvision import transforms

        test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        img_tensor = test_transforms(Image.fromarray(face_img))
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            features = self.model(img_tensor.to(self.device))
        features = features.data.cpu().numpy()
        return features

    def get_probab(self, features: Any) -> Any:
        import numpy as np

        x = np.dot(features, np.transpose(self.classifier_weights)) + self.classifier_bias
        return x

    def predict_emotions(self, face_img: Any, logits: bool = True) -> tuple[str, Any]:
        import numpy as np

        features = self.extract_features(face_img)
        scores = self.get_probab(features)[0]
        if self.is_mtl:
            x = scores[:-2]
        else:
            x = scores
        pred = int(np.argmax(x))
        return self.idx_to_class[pred], scores

def _get_emotion_classifier(
    model_name: str, device: int, dtype: torch.dtype
) -> Any:
    """Возвращает общий для процесса классификатор эмоций EfficientNet-B0."""
    key = (model_name, int(device), str(dtype))
    with _EFFICIENTNET_B0_LOCK:
        cached = _EFFICIENTNET_B0_CACHE.get(key)
        if cached is not None:
            return cached
        hf_device = "cuda" if device >= 0 and torch.cuda.is_available() else "cpu"
        try:
            clf = _EfficientNetB0EmotionRecognizer(model_name, hf_device)
            logger.info(
                "EfficientNet-B0 эмоции: %s — из локального кэша", model_name
            )
        except Exception:
            clf = _EfficientNetB0EmotionRecognizer(model_name, hf_device)
            logger.info(
                "EfficientNet-B0 эмоции: %s — первая загрузка весов", model_name
            )
        _EFFICIENTNET_B0_CACHE[key] = clf
        return clf


class EfficientNetB0EmotionClassifier:
    """Лёгкая обёртка над HSEmotionRecognizer с интерфейсом, совместимым с
    transformers pipeline image-classification, и прямым возвратом Emotion enum.

    __call__(face_rgb) -> list[{"label": str, "score": float}]
    classify_bgr(face_bgr) -> tuple[Emotion, float]
    """

    def __init__(
        self,
        model_name: str = EFFICIENTNET_B0_MODEL_NAME,
        device: int = -1,
        dtype: torch.dtype = torch.float32,
        emotion_conf_threshold: float = DEFAULT_EMOTION_CONF_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.emotion_conf_threshold = emotion_conf_threshold
        self._recognizer = _get_emotion_classifier(model_name, device, dtype)

    def __call__(self, face_rgb: np.ndarray) -> list[dict[str, Any]]:
        """Принимает RGB-изображение лица (numpy HWC), возвращает predictions в
        формате transformers pipeline: [{"label": ..., "score": ...}]."""
        emotion, confidence = self.classify_bgr(
            cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        )
        return [{"label": emotion.value, "score": confidence}]

    def classify_bgr(self, face_bgr: np.ndarray) -> tuple[Emotion, float]:
        """Принимает BGR-кроп лица от OpenCV, возвращает (Emotion, confidence)."""
        if face_bgr.size == 0:
            return Emotion.UNKNOWN, 0.0

        try:
            # HSEmotionRecognizer ожидает RGB numpy array для Image.fromarray.
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            label, scores = self._recognizer.predict_emotions(face_rgb, logits=True)
            probs = _softmax(np.asarray(scores, dtype=np.float32))
            best_idx = int(np.argmax(probs))
            best_conf = float(probs[best_idx])
            model_label = self._recognizer.idx_to_class.get(best_idx, label)
            emotion = _HSEMOTION_TO_EMOTION.get(model_label, Emotion.UNKNOWN)
            if best_conf < self.emotion_conf_threshold:
                emotion = Emotion.UNKNOWN
            return emotion, best_conf
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ошибка классификации эмоции EfficientNet-B0: %s", exc)
            return Emotion.UNKNOWN, 0.0


class EfficientNetB0EmotionModel(BaseFaceEmotionModel):
    """YOLOv8-face (детекция + ByteTrack) + EfficientNet-B0 (эмоции через hsemotion).

    Полный аналог CurrentYoloVitModel, но с классификатором эмоций EfficientNet-B0
    вместо ViT. Реализует контракт BaseFaceEmotionModel.
    """

    def __init__(
        self,
        model_path: str = "mood_base/model/yolov8n-face.pt",
        tracker_config: str = "configs/bytetrack.yaml",
        conf_threshold: float = 0.45,
        emotion_conf_threshold: float = DEFAULT_EMOTION_CONF_THRESHOLD,
        yolo_imgsz: int = 416,
        analyze_every_n_frames: int = 5,
        frame_scale: float = 0.5,
        batch_size: int = 8,
        cache_emotions: bool = True,
        emotion_model_name: str = EFFICIENTNET_B0_MODEL_NAME,
    ) -> None:
        # --- Выбор устройства (как в текущем пайплайне) ---
        if torch.cuda.is_available():
            self.device = 0
            self.yolo_device = 0
            self.torch_dtype = torch.float16
            logger.info("CUDA доступна: %s", torch.cuda.get_device_name(0))
        else:
            self.device = -1
            self.yolo_device = "cpu"
            self.torch_dtype = torch.float32
            logger.info("CUDA недоступна, используется CPU")

        # --- Параметры ---
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.emotion_conf_threshold = emotion_conf_threshold
        self.yolo_imgsz = yolo_imgsz
        self.analyze_every_n_frames = analyze_every_n_frames
        self.frame_scale = frame_scale
        self.batch_size = max(1, batch_size)
        self.cache_emotions = cache_emotions

        # --- Модели ---
        # YOLO — свой экземпляр на каждый объект (ByteTrack хранит треки).
        from ultralytics import YOLO

        self.face_model = YOLO(model_path)
        # EfficientNet-B0 классификатор эмоций — общий процессный кэш.
        self.emotion_classifier = EfficientNetB0EmotionClassifier(
            model_name=emotion_model_name,
            device=self.device,
            dtype=self.torch_dtype,
            emotion_conf_threshold=emotion_conf_threshold,
        )

        # --- Кэш эмоций по track_id ---
        self._last_results: dict[int, tuple[str, float]] = {}

    def _classify_emotion(self, face_bgr: np.ndarray) -> tuple[str, float]:
        emotion, confidence = self.emotion_classifier.classify_bgr(face_bgr)
        return emotion.value, float(confidence)

    def _process_one_frame(
        self, frame_id: int, frame: np.ndarray, result
    ) -> FrameResult:
        """Обрабатывает результат YOLO для одного кадра в FrameResult."""
        boxes = result.boxes
        detections: list[FaceDetection] = []

        current_ids: set[int] = set()
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().tolist()
            confidences = boxes.conf.tolist()
            bboxes = boxes.xyxy.int().tolist()

            h, w = frame.shape[:2]

            for track_id, face_conf, raw_bbox in zip(track_ids, confidences, bboxes):
                current_ids.add(track_id)
                # Возвращаем координаты к исходному разрешению.
                x1, y1, x2, y2 = raw_bbox
                x1 = int(x1 / self.frame_scale)
                y1 = int(y1 / self.frame_scale)
                x2 = int(x2 / self.frame_scale)
                y2 = int(y2 / self.frame_scale)

                # Обрезаем по границам кадра.
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Эмоцию считаем не каждый кадр (кэш по track_id).
                if frame_id % self.analyze_every_n_frames == 0:
                    face_crop = frame[y1:y2, x1:x2]
                    try:
                        emotion, emotion_conf = self._classify_emotion(face_crop)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Ошибка классификации эмоции для track_id=%s: %s",
                            track_id,
                            exc,
                        )
                        emotion, emotion_conf = "unknown", 0.0
                    if self.cache_emotions:
                        self._last_results[track_id] = (emotion, emotion_conf)
                else:
                    if self.cache_emotions:
                        emotion, emotion_conf = self._last_results.get(
                            track_id, ("unknown", 0.0)
                        )
                    else:
                        emotion, emotion_conf = "unknown", 0.0

                detections.append(
                    FaceDetection(
                        track_id=track_id,
                        bbox=BoundingBox(
                            x_min=x1, y_min=y1, x_max=x2, y_max=y2
                        ),
                        emotion=Emotion(emotion),
                        face_confidence=float(face_conf),
                        emotion_confidence=float(emotion_conf),
                    )
                )

        # Удаляем из кэша эмоции треков, которых нет в текущем кадре.
        for stale_tid in [tid for tid in self._last_results if tid not in current_ids]:
            del self._last_results[stale_tid]

        return FrameResult(frame_id=frame_id, detections=detections)

    def predict_batch(self, frames: list[tuple[int, np.ndarray]]) -> list[FrameResult]:
        """Обрабатывает пакет кадров одним батчевым вызовом YOLO."""
        if not frames:
            return []

        small_frames = [
            cv2.resize(
                f,
                None,
                fx=self.frame_scale,
                fy=self.frame_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            for _, f in frames
        ]

        try:
            results = self.face_model.track(
                small_frames,
                persist=True,
                tracker=self.tracker_config,
                conf=self.conf_threshold,
                imgsz=self.yolo_imgsz,
                device=self.yolo_device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Батч YOLO не удался (%s), перехожу на по-кадровую обработку", exc
            )
            return [self.predict(frame_id, frame) for frame_id, frame in frames]

        if not isinstance(results, list):
            results = [results]

        return [
            self._process_one_frame(frame_id, frame, res)
            for (frame_id, frame), res in zip(frames, results)
        ]

    def predict(self, frame_id: int, frame: np.ndarray) -> FrameResult:
        small = cv2.resize(
            frame,
            None,
            fx=self.frame_scale,
            fy=self.frame_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        results = self.face_model.track(
            small,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            imgsz=self.yolo_imgsz,
            device=self.yolo_device,
            verbose=False,
        )
        return self._process_one_frame(frame_id, frame, results[0])





