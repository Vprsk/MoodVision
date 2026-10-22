"""Трёхуровневый модуль детекции тела, позы и эмоций по языку тела (Сцены 2 и 3)."""

from collections import deque
import math
import threading
from typing import Any

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from ultralytics import YOLO

from app.core.logging import get_logger
from contracts.schemas import (
    BodyDetection,
    BodyPostureTag,
    BoundingBox,
    Emotion,
    Keypoint2D,
)

logger = get_logger("body_detect")

DEFAULT_BODY_EMOTION_MODEL_ID = "trpakov/vit-face-expression"

# Потокобезопасный процессный кэш для визуальной модели Уровня 3
_BODY_PIPELINE_CACHE: dict[tuple[str, int, str], Any] = {}
_BODY_PIPELINE_LOCK = threading.Lock()


def _build_body_pipeline(
    model_id: str,
    device: int,
    dtype: torch.dtype,
    *,
    offline: bool,
) -> Any:
    load_kwargs = {"local_files_only": offline}
    hf_model = AutoModelForImageClassification.from_pretrained(
        model_id, dtype=dtype, **load_kwargs
    )
    image_processor = AutoImageProcessor.from_pretrained(model_id, **load_kwargs)
    return pipeline(
        "image-classification",
        model=hf_model,
        image_processor=image_processor,
        device=device,
        dtype=dtype,
    )


def _get_body_emotion_pipeline(model_id: str, device: int, dtype: torch.dtype) -> Any:
    """Возвращает общий для процесса пайплайн визуальной оценки кропа тела (Уровень 3)."""
    key = (model_id, int(device), str(dtype))
    with _BODY_PIPELINE_LOCK:
        cached = _BODY_PIPELINE_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            clf = _build_body_pipeline(model_id, device, dtype, offline=True)
            logger.info("Body Emotion (HF): %s — из локального кэша", model_id)
        except Exception:
            clf = _build_body_pipeline(model_id, device, dtype, offline=False)
            logger.info("Body Emotion (HF): %s — первая загрузка с HF Hub", model_id)
        _BODY_PIPELINE_CACHE[key] = clf
        return clf


class YoloBodyPoseDetector:
    """Детектор тела, трекер и 3-уровневый классификатор языка тела на базе YOLOv8-pose."""

    # Индексы точек COCO
    NOSE = 0
    L_EAR, R_EAR = 3, 4
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12

    POSTURE_TO_EMOTION_PRIOR: dict[BodyPostureTag, dict[Emotion, float]] = {
        BodyPostureTag.AGITATED_GESTURES: {Emotion.ANGRY: 0.80, Emotion.SURPRISE: 0.15, Emotion.NEUTRAL: 0.05},
        BodyPostureTag.HANDS_ON_HIPS: {Emotion.ANGRY: 0.75, Emotion.DISGUST: 0.15, Emotion.NEUTRAL: 0.10},
        BodyPostureTag.ARMS_CROSSED: {Emotion.DISGUST: 0.50, Emotion.ANGRY: 0.35, Emotion.NEUTRAL: 0.15},
        BodyPostureTag.HANDS_TO_HEAD: {Emotion.FEAR: 0.45, Emotion.SAD: 0.35, Emotion.ANGRY: 0.20},
        BodyPostureTag.SLOUCHED_HEAD_DOWN: {Emotion.SAD: 0.80, Emotion.NEUTRAL: 0.20},
        BodyPostureTag.LEANING_FORWARD: {Emotion.ANGRY: 0.78, Emotion.NEUTRAL: 0.22},
        BodyPostureTag.NEUTRAL_POSTURE: {Emotion.NEUTRAL: 0.85, Emotion.HAPPY: 0.15},
    }

    def __init__(
        self,
        pose_model_path: str = "mood_base/model/yolov8n-pose.pt",
        body_emotion_model_id: str = DEFAULT_BODY_EMOTION_MODEL_ID,
        tracker_config: str = "configs/bytetrack.yaml",
        conf_threshold: float = 0.40,
        body_emotion_conf_threshold: float = 0.35,
        yolo_imgsz: int = 416,
        frame_scale: float = 0.5,
        analyze_every_n_frames: int = 5,
        device: int = 0,
        yolo_device: str | int = 0,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.body_emotion_conf_threshold = body_emotion_conf_threshold
        self.yolo_imgsz = yolo_imgsz
        self.frame_scale = frame_scale
        self.analyze_every_n_frames = analyze_every_n_frames
        self.device = device
        self.yolo_device = yolo_device
        self.torch_dtype = torch_dtype

        self.pose_model = YOLO(pose_model_path)
        self.body_classifier = _get_body_emotion_pipeline(
            body_emotion_model_id, self.device, self.torch_dtype
        )

        # История относительных координат запястий по track_id (Уровень 1)
        self._wrist_history: dict[int, deque[tuple[float, float, float, float]]] = {}
        # Кэш эмоций тела по track_id (Уровень 3)
        self._last_body_emotions: dict[int, tuple[Emotion, float]] = {}

    def detect(self, frame_id: int, frame: np.ndarray) -> list[BodyDetection]:
        """Запускает трекинг поз и 3-уровневую оценку языка тела."""
        h, w = frame.shape[:2]
        small = cv2.resize(
            frame,
            None,
            fx=self.frame_scale,
            fy=self.frame_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        results = self.pose_model.track(
            small,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            imgsz=self.yolo_imgsz,
            device=self.yolo_device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes
        kpts_data = result.keypoints

        bodies: list[BodyDetection] = []
        current_ids: set[int] = set()

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().tolist()
            confidences = boxes.conf.tolist()
            bboxes = boxes.xyxy.int().tolist()

            for idx, (track_id, body_conf, raw_bbox) in enumerate(
                zip(track_ids, confidences, bboxes)
            ):
                current_ids.add(track_id)
                x1, y1, x2, y2 = raw_bbox
                x1 = max(0, int(x1 / self.frame_scale))
                y1 = max(0, int(y1 / self.frame_scale))
                x2 = min(w, int(x2 / self.frame_scale))
                y2 = min(h, int(y2 / self.frame_scale))

                if x2 <= x1 or y2 <= y1:
                    continue

                bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

                keypoints: list[Keypoint2D] = []
                if kpts_data is not None and kpts_data.xy is not None:
                    xy_arr = kpts_data.xy[idx].cpu().numpy() / self.frame_scale
                    conf_arr = (
                        kpts_data.conf[idx].cpu().numpy()
                        if kpts_data.conf is not None
                        else np.ones(len(xy_arr))
                    )
                    keypoints = [
                        Keypoint2D(
                            x=float(pt[0]),
                            y=float(pt[1]),
                            conf=float(np.clip(c, 0.0, 1.0)),
                        )
                        for pt, c in zip(xy_arr, conf_arr)
                    ]

                # Уровень 1: Кинематика жестов
                kinetic_energy = self._compute_level1_kinetics(
                    track_id, bbox, keypoints
                )

                # Уровень 2: Биомеханика позы (BERA)
                posture_tag, pose_feature_probs = self._compute_level2_posture(
                    keypoints, kinetic_energy
                )

                # Уровень 3: Нейросеть по кропу фигуры + ансамбль с вектором позы
                if frame_id % self.analyze_every_n_frames == 0 or track_id not in self._last_body_emotions:
                    body_crop = frame[y1:y2, x1:x2]
                    body_em, body_em_conf = self._compute_level3_body_emotion(
                        body_crop, posture_tag, pose_feature_probs
                    )
                    self._last_body_emotions[track_id] = (body_em, body_em_conf)
                else:
                    body_em, body_em_conf = self._last_body_emotions[track_id]

                bodies.append(
                    BodyDetection(
                        track_id=track_id,
                        bbox=bbox,
                        body_confidence=float(body_conf),
                        keypoints=keypoints,
                        kinetic_energy=round(kinetic_energy, 4),
                        posture_tag=posture_tag,
                        body_emotion=body_em,
                        body_emotion_confidence=round(body_em_conf, 4),
                    )
                )

        # Очистка кэша для ушедших треков
        for stale_tid in [tid for tid in self._wrist_history if tid not in current_ids]:
            del self._wrist_history[stale_tid]
        for stale_tid in [tid for tid in self._last_body_emotions if tid not in current_ids]:
            del self._last_body_emotions[stale_tid]

        return bodies

    def _compute_level1_kinetics(
        self,
        track_id: int,
        bbox: BoundingBox,
        kpts: list[Keypoint2D],
    ) -> float:
        """Уровень 1: Вычисляет скорость движения запястий относительно центра плеч."""
        if len(kpts) < 11 or bbox.height <= 0:
            return 0.0

        lw, rw = kpts[self.L_WRIST], kpts[self.R_WRIST]
        ls, rs = kpts[self.L_SHOULDER], kpts[self.R_SHOULDER]

        if lw.conf < 0.25 and rw.conf < 0.25:
            return 0.0

        # Центр плеч (чтобы ходьба человека целиком не считалась взмахами рук)
        cx = 0.5 * (ls.x + rs.x) if (ls.conf > 0.2 and rs.conf > 0.2) else bbox.center_x
        cy = 0.5 * (ls.y + rs.y) if (ls.conf > 0.2 and rs.conf > 0.2) else bbox.center_y

        rel_lwx = (lw.x - cx) if lw.conf >= 0.25 else 0.0
        rel_lwy = (lw.y - cy) if lw.conf >= 0.25 else 0.0
        rel_rwx = (rw.x - cx) if rw.conf >= 0.25 else 0.0
        rel_rwy = (rw.y - cy) if rw.conf >= 0.25 else 0.0

        hist = self._wrist_history.setdefault(track_id, deque(maxlen=10))
        hist.append((rel_lwx, rel_lwy, rel_rwx, rel_rwy))

        if len(hist) < 2:
            return 0.0

        diffs = np.diff(np.array(hist), axis=0)
        speed = float(
            np.mean(
                np.hypot(diffs[:, 0], diffs[:, 1])
                + np.hypot(diffs[:, 2], diffs[:, 3])
            )
        )
        norm_speed = speed / float(max(bbox.height, 40))
        return float(np.clip(norm_speed * 4.5, 0.0, 1.0))

    def _compute_level2_posture(
        self,
        kpts: list[Keypoint2D],
        kinetic_energy: float,
    ) -> tuple[BodyPostureTag, dict[Emotion, float]]:
        """Уровень 2: Биомеханическая классификация позы и расчёт угловых пропорций скелета."""
        if len(kpts) < 13:
            tag = (
                BodyPostureTag.AGITATED_GESTURES
                if kinetic_energy > 0.38
                else BodyPostureTag.NEUTRAL_POSTURE
            )
            return tag, dict(self.POSTURE_TO_EMOTION_PRIOR[tag])

        nose = kpts[self.NOSE]
        ls, rs = kpts[self.L_SHOULDER], kpts[self.R_SHOULDER]
        le, re = kpts[self.L_ELBOW], kpts[self.R_ELBOW]
        lw, rw = kpts[self.L_WRIST], kpts[self.R_WRIST]
        lh, rh = kpts[self.L_HIP], kpts[self.R_HIP]

        shoulder_w = max(math.hypot(ls.x - rs.x, ls.y - rs.y), 20.0)
        mid_shoulder_x = 0.5 * (ls.x + rs.x)
        mid_shoulder_y = 0.5 * (ls.y + rs.y)
        mid_hip_x = 0.5 * (lh.x + rh.x)
        mid_hip_y = 0.5 * (lh.y + rh.y)
        torso_h = max(math.hypot(mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y), 30.0)

        # 1. Руки у головы / лица ("схватился за голову" — стресс/испуг/возмущение)
        if nose.conf > 0.3:
            dist_lw_head = math.hypot(lw.x - nose.x, lw.y - nose.y) if lw.conf > 0.3 else 9999.0
            dist_rw_head = math.hypot(rw.x - nose.x, rw.y - nose.y) if rw.conf > 0.3 else 9999.0
            if min(dist_lw_head, dist_rw_head) < 0.65 * shoulder_w:
                tag = BodyPostureTag.HANDS_TO_HEAD
                return tag, dict(self.POSTURE_TO_EMOTION_PRIOR[tag])

        # 2. Возмущённые взмахи руками или вскинутые выше плеч руки
        hands_above_shoulders = (
            (lw.conf > 0.35 and ls.conf > 0.35 and lw.y < ls.y)
            or (rw.conf > 0.35 and rs.conf > 0.35 and rw.y < rs.y)
        )
        if kinetic_energy > 0.38 or hands_above_shoulders:
            tag = BodyPostureTag.AGITATED_GESTURES
            return tag, dict(self.POSTURE_TO_EMOTION_PRIOR[tag])

        # 3. Скрещенные на груди руки (закрытая поза недовольства)
        if lw.conf > 0.35 and rw.conf > 0.35 and le.conf > 0.3 and re.conf > 0.3:
            in_chest_y = (
                mid_shoulder_y < lw.y < mid_hip_y
                and mid_shoulder_y < rw.y < mid_hip_y
            )
            dist_lw_re = math.hypot(lw.x - re.x, lw.y - re.y)
            dist_rw_le = math.hypot(rw.x - le.x, rw.y - le.y)
            if in_chest_y and (dist_lw_re < 0.75 * shoulder_w or dist_rw_le < 0.75 * shoulder_w):
                tag = BodyPostureTag.ARMS_CROSSED
                return tag, dict(self.POSTURE_TO_EMOTION_PRIOR[tag])

        # 4. Руки в боки (вызов / претензия в очереди)
        if lw.conf > 0.35 and rw.conf > 0.35 and lh.conf > 0.3 and rh.conf > 0.3:
            dist_lw_lh = math.hypot(lw.x - lh.x, lw.y - lh.y)
            dist_rw_rh = math.hypot(rw.x - rh.x, rw.y - rh.y)
            elbows_wide = abs(le.x - re.x) > 1.25 * shoulder_w
            if dist_lw_lh < 0.65 * shoulder_w and dist_rw_rh < 0.65 * shoulder_w and elbows_wide:
                tag = BodyPostureTag.HANDS_ON_HIPS
                return tag, dict(self.POSTURE_TO_EMOTION_PRIOR[tag])

        # 5. Наклон корпуса вперёд (нависание над стойкой во время спора)
        if torso_h / shoulder_w < 0.85 and kinetic_energy > 0.22:
            tag = BodyPostureTag.LEANING_FORWARD
            return tag, dict(self.POSTURE_TO_EMOTION_PRIOR[tag])

        # 6. Опущенная голова и сутулость (подавленность / грусть)
        if nose.conf > 0.35 and ls.conf > 0.35 and rs.conf > 0.35:
            head_neck_dist = mid_shoulder_y - nose.y
            if head_neck_dist < 0.25 * torso_h and kinetic_energy < 0.15:
                tag = BodyPostureTag.SLOUCHED_HEAD_DOWN
                return tag, dict(self.POSTURE_TO_EMOTION_PRIOR[tag])

        tag = BodyPostureTag.NEUTRAL_POSTURE
        return tag, dict(self.POSTURE_TO_EMOTION_PRIOR[tag])

    def _compute_level3_body_emotion(
        self,
        body_crop_bgr: np.ndarray,
        posture_tag: BodyPostureTag,
        pose_priors: dict[Emotion, float],
    ) -> tuple[Emotion, float]:
        """Уровень 3: Ансамбль визуальной нейросети по кропу тела и биомеханического вектора скелета."""
        combined_scores: dict[Emotion, float] = {em: 0.0 for em in Emotion}

        # Вклад биомеханического вектора скелета (вес 0.55)
        for em, prob in pose_priors.items():
            combined_scores[em] += 0.55 * prob

        # Вклад визуальной нейросети по кропу фигуры человека (вес 0.45)
        if body_crop_bgr.size > 0:
            try:
                crop_rgb = cv2.cvtColor(body_crop_bgr, cv2.COLOR_BGR2RGB)
                preds = self.body_classifier(crop_rgb)
                for item in preds:
                    label_str = item["label"].lower()
                    score = float(item["score"])
                    try:
                        em = Emotion(label_str)
                        combined_scores[em] += 0.45 * score
                    except ValueError:
                        continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ошибка классификатора тела Уровня 3: %s", exc)

        best_emotion = max(combined_scores, key=lambda k: combined_scores[k])
        best_conf = float(np.clip(combined_scores[best_emotion], 0.0, 1.0))

        # Если поза явно не нейтральная, гарантируем минимальную уверенность
        if posture_tag != BodyPostureTag.NEUTRAL_POSTURE and best_conf < self.body_emotion_conf_threshold:
            prior_best = max(pose_priors, key=lambda k: pose_priors[k])
            return prior_best, max(best_conf, 0.60)

        if best_conf < self.body_emotion_conf_threshold:
            return Emotion.UNKNOWN, best_conf

        return best_emotion, best_conf