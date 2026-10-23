"""Реализация мультимодальных пайплайнов для Сцены 1, Сцены 2 и Сцены 3."""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.core.logging import get_logger
from contracts.base import BaseFaceEmotionModel
from contracts.schemas import (
    BoundingBox,
    Emotion,
    FaceDetection,
    FrameResult,
)
from mood_base.audio_branch import DEFAULT_AUDIO_MODEL_ID, AudioAffectAnalyzer
from mood_base.body_detect import (
    DEFAULT_BODY_EMOTION_MODEL_ID,
    YoloBodyPoseDetector,
)
from mood_base.fusion_engine import MultimodalFusionEngine
from mood_base.models.current_yolo_vit import (
    EMOTION_MODEL_ID,
    _get_emotion_pipeline,
)

logger = get_logger("multimodal_scenes")


def _select_device() -> tuple[int, str | int, torch.dtype]:
    if torch.cuda.is_available():
        return 0, 0, torch.float16
    return -1, "cpu", torch.float32


class Scene1FaceAudioModel(BaseFaceEmotionModel):
    """Сцена 1: Лицо (YOLOv8-face + ByteTrack + Quality Gate + ViT) + Звук."""

    def __init__(
        self,
        model_path: str = "mood_base/model/yolov8n-face.pt",
        audio_model_id: str = DEFAULT_AUDIO_MODEL_ID,
        tracker_config: str = "configs/bytetrack.yaml",
        conf_threshold: float = 0.45,
        emotion_conf_threshold: float = 0.35,
        min_face_px: int = 48,
        alert_threshold: float = 0.65,
        temporal_window: int = 15,
        yolo_imgsz: int = 416,
        analyze_every_n_frames: int = 5,
        frame_scale: float = 0.5,
    ) -> None:
        self.device, self.yolo_device, self.torch_dtype = _select_device()
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.emotion_conf_threshold = emotion_conf_threshold
        self.yolo_imgsz = yolo_imgsz
        self.analyze_every_n_frames = analyze_every_n_frames
        self.frame_scale = frame_scale

        self.face_model = YOLO(model_path)
        self.emotion_classifier = _get_emotion_pipeline(
            EMOTION_MODEL_ID, self.device, self.torch_dtype
        )
        self.audio_analyzer = AudioAffectAnalyzer(
            model_id=audio_model_id,
            device=self.device,
            torch_dtype=self.torch_dtype,
            lazy_load=True,
        )
        self.fusion = MultimodalFusionEngine(
            min_face_px=min_face_px,
            temporal_window=temporal_window,
            alert_threshold=alert_threshold,
        )
        self._last_results: dict[int, tuple[Emotion, float]] = {}

    def _classify_emotion(self, face_bgr: np.ndarray) -> tuple[Emotion, float]:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        predictions = self.emotion_classifier(face_rgb)
        best = predictions[0]
        confidence = float(best["score"])
        if confidence < self.emotion_conf_threshold:
            return Emotion.UNKNOWN, confidence
        return Emotion(best["label"].lower()), confidence

    def predict(
        self,
        frame_id: int,
        frame: np.ndarray,
        audio_chunk: np.ndarray | None = None,
    ) -> FrameResult:
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
        boxes = results[0].boxes
        detections: list[FaceDetection] = []
        current_ids: set[int] = set()

        if boxes is not None and boxes.id is not None:
            h, w = frame.shape[:2]
            for track_id, face_conf, raw_bbox in zip(
                boxes.id.int().tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.int().tolist(),
            ):
                current_ids.add(track_id)
                x1 = max(0, int(raw_bbox[0] / self.frame_scale))
                y1 = max(0, int(raw_bbox[1] / self.frame_scale))
                x2 = min(w, int(raw_bbox[2] / self.frame_scale))
                y2 = min(h, int(raw_bbox[3] / self.frame_scale))
                if x2 <= x1 or y2 <= y1:
                    continue

                bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
                passed_gate = self.fusion.check_face_quality(frame, bbox)

                if not passed_gate:
                    emotion, emotion_conf = Emotion.UNKNOWN, 0.0
                elif (
                    frame_id % self.analyze_every_n_frames == 0
                    or track_id not in self._last_results
                ):
                    try:
                        emotion, emotion_conf = self._classify_emotion(
                            frame[y1:y2, x1:x2]
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Ошибка классификации лица (track=%s): %s",
                            track_id,
                            exc,
                        )
                        emotion, emotion_conf = Emotion.UNKNOWN, 0.0
                    self._last_results[track_id] = (emotion, emotion_conf)
                else:
                    emotion, emotion_conf = self._last_results[track_id]

                detections.append(
                    FaceDetection(
                        track_id=track_id,
                        bbox=bbox,
                        emotion=emotion,
                        face_confidence=float(face_conf),
                        emotion_confidence=float(emotion_conf),
                        passed_quality_gate=passed_gate,
                    )
                )

        for stale_tid in [
            tid for tid in self._last_results if tid not in current_ids
        ]:
            del self._last_results[stale_tid]

        # Анализ звука и слияние модальностей (Лицо + Звук)
        audio_res = self.audio_analyzer.analyze(audio_chunk)
        track_events = self.fusion.fuse_scene1(detections, audio_res)
        events_by_tid = {ev.track_id: ev for ev in track_events}

        # Прокидываем итоговую мультимодальную эмоцию в detections для отрисовки в UI
        updated_detections: list[FaceDetection] = []
        for det in detections:
            ev = (
                events_by_tid.get(det.track_id)
                if det.track_id is not None
                else None
            )
            if ev is not None and (
                ev.dominant_modality == "audio"
                or ev.is_negative_alert
                or det.emotion == Emotion.UNKNOWN
            ):
                det = det.model_copy(
                    update={
                        "emotion": ev.final_emotion,
                        "emotion_confidence": round(
                            max(det.emotion_confidence, ev.negative_score), 4
                        ),
                    }
                )
            updated_detections.append(det)

        return FrameResult(
            frame_id=frame_id,
            detections=updated_detections,
            audio=audio_res,
            track_events=track_events,
        )


class Scene2FaceBodyAudioModel(BaseFaceEmotionModel):
    """Сцена 2: Тело (трекинг + 3 уровня языка тела) + Лицо (Quality Gate + ViT) + Звук."""

    def __init__(
        self,
        face_model_path: str = "mood_base/model/yolov8n-face.pt",
        pose_model_path: str = "mood_base/model/yolov8n-pose.pt",
        body_emotion_model_id: str = DEFAULT_BODY_EMOTION_MODEL_ID,
        audio_model_id: str = DEFAULT_AUDIO_MODEL_ID,
        tracker_config: str = "configs/bytetrack.yaml",
        conf_threshold: float = 0.45,
        body_conf_threshold: float = 0.40,
        emotion_conf_threshold: float = 0.35,
        min_face_px: int = 64,
        alert_threshold: float = 0.65,
        temporal_window: int = 15,
        yolo_imgsz: int = 416,
        analyze_every_n_frames: int = 5,
        frame_scale: float = 0.5,
    ) -> None:
        self.device, self.yolo_device, self.torch_dtype = _select_device()
        self.conf_threshold = conf_threshold
        self.emotion_conf_threshold = emotion_conf_threshold
        self.yolo_imgsz = yolo_imgsz
        self.analyze_every_n_frames = analyze_every_n_frames
        self.frame_scale = frame_scale

        # Ведущий детектор и трекер — по ТЕЛУ (3 уровня языка тела)
        self.body_detector = YoloBodyPoseDetector(
            pose_model_path=pose_model_path,
            body_emotion_model_id=body_emotion_model_id,
            tracker_config=tracker_config,
            conf_threshold=body_conf_threshold,
            body_emotion_conf_threshold=emotion_conf_threshold,
            yolo_imgsz=yolo_imgsz,
            frame_scale=frame_scale,
            analyze_every_n_frames=analyze_every_n_frames,
            device=self.device,
            yolo_device=self.yolo_device,
            torch_dtype=self.torch_dtype,
        )

        # Детектор лица работает без собственного трекера (привязывается к телу)
        self.face_model = YOLO(face_model_path)
        self.emotion_classifier = _get_emotion_pipeline(
            EMOTION_MODEL_ID, self.device, self.torch_dtype
        )
        self.audio_analyzer = AudioAffectAnalyzer(
            model_id=audio_model_id,
            device=self.device,
            torch_dtype=self.torch_dtype,
            lazy_load=True,
        )
        self.fusion = MultimodalFusionEngine(
            min_face_px=min_face_px,
            temporal_window=temporal_window,
            alert_threshold=alert_threshold,
        )
        self._last_face_emotions: dict[int, tuple[Emotion, float]] = {}

    def _detect_raw_faces(self, frame: np.ndarray) -> list[FaceDetection]:
        h, w = frame.shape[:2]
        small = cv2.resize(
            frame,
            None,
            fx=self.frame_scale,
            fy=self.frame_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        results = self.face_model.predict(
            small,
            conf=self.conf_threshold,
            imgsz=self.yolo_imgsz,
            device=self.yolo_device,
            verbose=False,
        )
        boxes = results[0].boxes
        raw_faces: list[FaceDetection] = []
        if boxes is not None:
            for face_conf, raw_bbox in zip(
                boxes.conf.tolist(), boxes.xyxy.int().tolist()
            ):
                x1 = max(0, int(raw_bbox[0] / self.frame_scale))
                y1 = max(0, int(raw_bbox[1] / self.frame_scale))
                x2 = min(w, int(raw_bbox[2] / self.frame_scale))
                y2 = min(h, int(raw_bbox[3] / self.frame_scale))
                if x2 <= x1 or y2 <= y1:
                    continue
                raw_faces.append(
                    FaceDetection(
                        track_id=None,
                        bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                        emotion=Emotion.UNKNOWN,
                        face_confidence=float(face_conf),
                        emotion_confidence=0.0,
                    )
                )
        return raw_faces

    def _classify_face_crop(
        self, face_bgr: np.ndarray
    ) -> tuple[Emotion, float]:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        predictions = self.emotion_classifier(face_rgb)
        best = predictions[0]
        confidence = float(best["score"])
        if confidence < self.emotion_conf_threshold:
            return Emotion.UNKNOWN, confidence
        return Emotion(best["label"].lower()), confidence

    def predict(
        self,
        frame_id: int,
        frame: np.ndarray,
        audio_chunk: np.ndarray | None = None,
    ) -> FrameResult:
        # 1. Трекинг тел и расчёт 3 уровней языка тела
        bodies = self.body_detector.detect(frame_id, frame)

        # 2. Детекция лиц и геометрическая сшивка с телами
        raw_faces = self._detect_raw_faces(frame)
        matched_faces = self.fusion.match_faces_to_bodies(raw_faces, bodies)

        # 3. Условная классификация мимики только для лиц, прошедших Quality Gate
        active_body_ids = {b.track_id for b in bodies if b.track_id is not None}
        for face in matched_faces:
            passed_gate = self.fusion.check_face_quality(frame, face.bbox)
            face.passed_quality_gate = passed_gate
            if not passed_gate:
                face.emotion = Emotion.UNKNOWN
                face.emotion_confidence = 0.0
                continue

            tid = face.track_id
            if (
                tid is None
                or frame_id % self.analyze_every_n_frames == 0
                or tid not in self._last_face_emotions
            ):
                crop = frame[
                    face.bbox.y_min : face.bbox.y_max,
                    face.bbox.x_min : face.bbox.x_max,
                ]
                try:
                    em, conf = self._classify_face_crop(crop)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Ошибка классификации лица в Сцене 2: %s", exc
                    )
                    em, conf = Emotion.UNKNOWN, 0.0
                if tid is not None:
                    self._last_face_emotions[tid] = (em, conf)
            else:
                em, conf = self._last_face_emotions[tid]

            face.emotion = em
            face.emotion_confidence = float(conf)

        for stale_tid in [
            tid
            for tid in self._last_face_emotions
            if tid not in active_body_ids
        ]:
            del self._last_face_emotions[stale_tid]

        # 4. Анализ звука и слияние модальностей (Лицо + Звук + Тело)
        audio_res = self.audio_analyzer.analyze(audio_chunk)
        track_events = self.fusion.fuse_scene2(bodies, audio_res)
        events_by_tid = {ev.track_id: ev for ev in track_events}

        # 5. Формирование списка detections для совместимости с UI-плеером и try.py
        ui_detections: list[FaceDetection] = []
        for body in bodies:
            ev = (
                events_by_tid.get(body.track_id)
                if body.track_id is not None
                else None
            )
            if body.face is not None and body.face.passed_quality_gate:
                face_det = body.face
                # Если звук или поза перевесили спокойное лицо (или сработал алерт), обновляем эмоцию
                if ev is not None and (
                    ev.dominant_modality in ("audio", "body")
                    or ev.is_negative_alert
                    or face_det.emotion == Emotion.UNKNOWN
                ):
                    face_det = face_det.model_copy(
                        update={
                            "emotion": ev.final_emotion,
                            "emotion_confidence": round(
                                max(
                                    face_det.emotion_confidence,
                                    ev.negative_score,
                                ),
                                4,
                            ),
                        }
                    )
                ui_detections.append(face_det)
            else:
                # Человек стоит спиной или лицо мелкое: отдаём рамку и эмоцию тела/звука
                final_em = (
                    ev.final_emotion if ev is not None else body.body_emotion
                )
                final_conf = (
                    max(body.body_emotion_confidence, ev.negative_score)
                    if ev is not None
                    else body.body_emotion_confidence
                )
                ui_detections.append(
                    FaceDetection(
                        track_id=body.track_id,
                        bbox=body.bbox,
                        emotion=final_em,
                        face_confidence=body.body_confidence,
                        emotion_confidence=round(float(final_conf), 4),
                        passed_quality_gate=False,
                    )
                )

        return FrameResult(
            frame_id=frame_id,
            detections=ui_detections,
            bodies=bodies,
            audio=audio_res,
            track_events=track_events,
        )


class Scene3BodyAudioModel(BaseFaceEmotionModel):
    """Сцена 3: Звук + Тело (3 уровня языка тела, без загрузки моделей лица)."""

    def __init__(
        self,
        pose_model_path: str = "mood_base/model/yolov8n-pose.pt",
        body_emotion_model_id: str = DEFAULT_BODY_EMOTION_MODEL_ID,
        audio_model_id: str = DEFAULT_AUDIO_MODEL_ID,
        tracker_config: str = "configs/bytetrack.yaml",
        body_conf_threshold: float = 0.40,
        emotion_conf_threshold: float = 0.35,
        alert_threshold: float = 0.65,
        temporal_window: int = 15,
        yolo_imgsz: int = 416,
        analyze_every_n_frames: int = 5,
        frame_scale: float = 0.5,
    ) -> None:
        self.device, self.yolo_device, self.torch_dtype = _select_device()

        self.body_detector = YoloBodyPoseDetector(
            pose_model_path=pose_model_path,
            body_emotion_model_id=body_emotion_model_id,
            tracker_config=tracker_config,
            conf_threshold=body_conf_threshold,
            body_emotion_conf_threshold=emotion_conf_threshold,
            yolo_imgsz=yolo_imgsz,
            frame_scale=frame_scale,
            analyze_every_n_frames=analyze_every_n_frames,
            device=self.device,
            yolo_device=self.yolo_device,
            torch_dtype=self.torch_dtype,
        )
        self.audio_analyzer = AudioAffectAnalyzer(
            model_id=audio_model_id,
            device=self.device,
            torch_dtype=self.torch_dtype,
            lazy_load=True,
        )
        self.fusion = MultimodalFusionEngine(
            temporal_window=temporal_window,
            alert_threshold=alert_threshold,
        )

    def predict(
        self,
        frame_id: int,
        frame: np.ndarray,
        audio_chunk: np.ndarray | None = None,
    ) -> FrameResult:
        bodies = self.body_detector.detect(frame_id, frame)
        audio_res = self.audio_analyzer.analyze(audio_chunk)
        track_events = self.fusion.fuse_scene3(bodies, audio_res)
        events_by_tid = {ev.track_id: ev for ev in track_events}

        # Прокси-детекции в FrameResult.detections для штатной отрисовки в UI и try.py
        ui_detections: list[FaceDetection] = []
        for body in bodies:
            ev = (
                events_by_tid.get(body.track_id)
                if body.track_id is not None
                else None
            )
            final_em = ev.final_emotion if ev is not None else body.body_emotion
            final_conf = (
                max(body.body_emotion_confidence, ev.negative_score)
                if ev is not None
                else body.body_emotion_confidence
            )
            ui_detections.append(
                FaceDetection(
                    track_id=body.track_id,
                    bbox=body.bbox,
                    emotion=final_em,
                    face_confidence=body.body_confidence,
                    emotion_confidence=round(float(final_conf), 4),
                    passed_quality_gate=False,
                )
            )

        return FrameResult(
            frame_id=frame_id,
            detections=ui_detections,
            bodies=bodies,
            audio=audio_res,
            track_events=track_events,
        )