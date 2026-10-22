"""Модуль адаптивного слияния модальностей (Quality-Gated Fusion) и временной агрегации."""

from collections import Counter, deque

import cv2
import numpy as np

from contracts.schemas import (
    AudioSegmentResult,
    BodyDetection,
    BodyPostureTag,
    BoundingBox,
    Emotion,
    FaceDetection,
    TrackAffectState,
)

NEGATIVE_EMOTIONS: set[Emotion] = {
    Emotion.ANGRY,
    Emotion.SAD,
    Emotion.FEAR,
    Emotion.DISGUST,
}


class MultimodalFusionEngine:
    """Связывает Лицо, Тело и Звук вокруг track_id и сглаживает решения во временном окне."""

    def __init__(
        self,
        min_face_px: int = 64,
        min_laplacian_var: float = 30.0,
        temporal_window: int = 15,
        alert_threshold: float = 0.65,
    ) -> None:
        self.min_face_px = min_face_px
        self.min_laplacian_var = min_laplacian_var
        self.temporal_window = max(1, temporal_window)
        self.alert_threshold = alert_threshold

        self._score_history: dict[int, deque[float]] = {}
        self._emotion_history: dict[int, deque[Emotion]] = {}

    def check_face_quality(
        self, frame: np.ndarray, bbox: BoundingBox
    ) -> bool:
        """Quality Gate: проверяет, достаточно ли крупное и чёткое лицо для запуска ViT."""
        if bbox.height < self.min_face_px or bbox.width < int(self.min_face_px * 0.75):
            return False

        crop = frame[bbox.y_min : bbox.y_max, bbox.x_min : bbox.x_max]
        if crop.size == 0:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return lap_var >= self.min_laplacian_var

    @staticmethod
    def match_faces_to_bodies(
        faces: list[FaceDetection],
        bodies: list[BodyDetection],
    ) -> list[FaceDetection]:
        """Геометрически привязывает найденные лица к телам (Сцена 2) и наследует track_id тела."""
        matched_faces: list[FaceDetection] = []
        used_body_ids: set[int] = set()

        for face in faces:
            cx, cy = face.bbox.center_x, face.bbox.center_y
            best_body: BodyDetection | None = None
            best_dist = float("inf")

            for idx, body in enumerate(bodies):
                if idx in used_body_ids:
                    continue
                upper_limit_y = body.bbox.y_min + 0.55 * body.bbox.height
                if (
                    body.bbox.x_min <= cx <= body.bbox.x_max
                    and body.bbox.y_min <= cy <= upper_limit_y
                ):
                    dist = abs(cx - body.bbox.center_x) + abs(cy - body.bbox.y_min)
                    if dist < best_dist:
                        best_dist = dist
                        best_body = body
                        best_body_idx = idx

            if best_body is not None and best_body.track_id is not None:
                used_body_ids.add(best_body_idx)
                face = face.model_copy(update={"track_id": best_body.track_id})
                best_body.face = face

            matched_faces.append(face)

        return matched_faces

    def fuse_scene1(
        self,
        faces: list[FaceDetection],
        audio: AudioSegmentResult | None,
    ) -> list[TrackAffectState]:
        """Слияние для Сцены 1 (Лицо + Звук)."""
        a_score, a_em = self._extract_audio_affect(audio)
        events: list[TrackAffectState] = []
        active_ids: set[int] = set()

        for face in faces:
            if face.track_id is None:
                continue
            tid = face.track_id
            active_ids.add(tid)

            f_score = (
                face.emotion_confidence
                if (face.passed_quality_gate and face.emotion in NEGATIVE_EMOTIONS)
                else 0.0
            )
            if audio is not None and (audio.speech_detected or audio.is_loud_spike):
                inst_score = 0.65 * f_score + 0.35 * a_score
            else:
                inst_score = f_score

            dom_mod = "face" if f_score >= a_score else "audio"
            inst_em = (
                face.emotion
                if (face.emotion != Emotion.UNKNOWN and f_score >= a_score)
                else (a_em if a_em != Emotion.UNKNOWN else face.emotion)
            )

            events.append(self._update_track_window(tid, inst_score, inst_em, dom_mod))

        self._cleanup_stale_tracks(active_ids)
        return events

    def fuse_scene2(
        self,
        bodies: list[BodyDetection],
        audio: AudioSegmentResult | None,
    ) -> list[TrackAffectState]:
        """Quality-Gated слияние для Сцены 2 (Лицо + Звук + Тело)."""
        a_score, a_em = self._extract_audio_affect(audio)
        most_active_tid = self._find_most_active_body_track(bodies)
        events: list[TrackAffectState] = []
        active_ids: set[int] = set()

        for body in bodies:
            if body.track_id is None:
                continue
            tid = body.track_id
            active_ids.add(tid)

            b_score = self._compute_body_negative_score(body)
            track_a_score = a_score if tid == most_active_tid else 0.3 * a_score

            face = body.face
            has_valid_face = (
                face is not None
                and face.passed_quality_gate
                and face.emotion != Emotion.UNKNOWN
            )

            if has_valid_face and face is not None:
                f_score = (
                    face.emotion_confidence
                    if face.emotion in NEGATIVE_EMOTIONS
                    else 0.0
                )
                inst_score = 0.50 * f_score + 0.25 * b_score + 0.25 * track_a_score
                scores_map = {
                    "face": (f_score, face.emotion),
                    "body": (b_score, body.body_emotion),
                    "audio": (track_a_score, a_em),
                }
            else:
                # Человек отвернулся или далеко: вес лица = 0, трек ведётся по Телу + Звуку
                inst_score = 0.55 * b_score + 0.45 * track_a_score
                scores_map = {
                    "body": (b_score, body.body_emotion),
                    "audio": (track_a_score, a_em),
                }

            dom_mod = max(scores_map, key=lambda k: scores_map[k][0])
            inst_em = scores_map[dom_mod][1]
            if inst_em == Emotion.UNKNOWN:
                inst_em = body.body_emotion if body.body_emotion != Emotion.UNKNOWN else Emotion.NEUTRAL

            events.append(self._update_track_window(tid, inst_score, inst_em, dom_mod))

        self._cleanup_stale_tracks(active_ids)
        return events

    def fuse_scene3(
        self,
        bodies: list[BodyDetection],
        audio: AudioSegmentResult | None,
    ) -> list[TrackAffectState]:
        """Слияние для Сцены 3 (Звук + Тело, без детектора лиц)."""
        a_score, a_em = self._extract_audio_affect(audio)
        most_active_tid = self._find_most_active_body_track(bodies)
        max_kinetics = max((b.kinetic_energy for b in bodies), default=0.0)
        any_non_neutral_posture = any(
            b.posture_tag != BodyPostureTag.NEUTRAL_POSTURE for b in bodies
        )

        # Защита от бытового шума: если все люди в кадре неподвижны и в спокойной позе
        if a_score > 0.6 and max_kinetics < 0.12 and not any_non_neutral_posture:
            a_score *= 0.30

        events: list[TrackAffectState] = []
        active_ids: set[int] = set()

        for body in bodies:
            if body.track_id is None:
                continue
            tid = body.track_id
            active_ids.add(tid)

            b_score = self._compute_body_negative_score(body)
            track_a_score = a_score if tid == most_active_tid else 0.25 * a_score

            inst_score = 0.55 * b_score + 0.45 * track_a_score
            dom_mod = "body" if b_score >= track_a_score else "audio"
            inst_em = (
                body.body_emotion
                if (dom_mod == "body" or a_em == Emotion.UNKNOWN)
                else a_em
            )
            if inst_em == Emotion.UNKNOWN:
                inst_em = Emotion.NEUTRAL

            events.append(self._update_track_window(tid, inst_score, inst_em, dom_mod))

        self._cleanup_stale_tracks(active_ids)
        return events

    @staticmethod
    def _extract_audio_affect(
        audio: AudioSegmentResult | None,
    ) -> tuple[float, Emotion]:
        if audio is None or (not audio.speech_detected and not audio.is_loud_spike):
            return 0.0, Emotion.UNKNOWN
        score = (
            audio.audio_confidence
            if audio.audio_emotion in NEGATIVE_EMOTIONS
            else 0.0
        )
        if audio.is_loud_spike:
            score = max(score, 0.80)
        em = (
            audio.audio_emotion
            if audio.audio_emotion != Emotion.UNKNOWN
            else (Emotion.ANGRY if audio.is_loud_spike else Emotion.UNKNOWN)
        )
        return score, em

    @staticmethod
    def _compute_body_negative_score(body: BodyDetection) -> float:
        base = (
            body.body_emotion_confidence
            if body.body_emotion in NEGATIVE_EMOTIONS
            else 0.0
        )
        if body.posture_tag in (
            BodyPostureTag.AGITATED_GESTURES,
            BodyPostureTag.HANDS_ON_HIPS,
            BodyPostureTag.LEANING_FORWARD,
        ):
            base = max(base, 0.70, body.kinetic_energy)
        elif body.posture_tag in (
            BodyPostureTag.ARMS_CROSSED,
            BodyPostureTag.HANDS_TO_HEAD,
            BodyPostureTag.SLOUCHED_HEAD_DOWN,
        ):
            base = max(base, 0.55)
        return float(np.clip(base, 0.0, 1.0))

    @staticmethod
    def _find_most_active_body_track(bodies: list[BodyDetection]) -> int | None:
        if not bodies:
            return None
        best = max(
            bodies,
            key=lambda b: (
                b.kinetic_energy
                + (0.35 if b.posture_tag != BodyPostureTag.NEUTRAL_POSTURE else 0.0)
            ),
        )
        return best.track_id

    def _update_track_window(
        self,
        track_id: int,
        inst_score: float,
        inst_emotion: Emotion,
        dominant_modality: str,
    ) -> TrackAffectState:
        s_buf = self._score_history.setdefault(
            track_id, deque(maxlen=self.temporal_window)
        )
        e_buf = self._emotion_history.setdefault(
            track_id, deque(maxlen=self.temporal_window)
        )

        s_buf.append(float(np.clip(inst_score, 0.0, 1.0)))
        if inst_emotion != Emotion.UNKNOWN:
            e_buf.append(inst_emotion)

        smoothed_score = float(np.mean(s_buf))
        is_alert = smoothed_score >= self.alert_threshold

        if is_alert:
            neg_emotions = [e for e in e_buf if e in NEGATIVE_EMOTIONS]
            final_em = (
                Counter(neg_emotions).most_common(1)[0][0]
                if neg_emotions
                else Emotion.ANGRY
            )
        else:
            final_em = (
                Counter(e_buf).most_common(1)[0][0] if e_buf else inst_emotion
            )

        return TrackAffectState(
            track_id=track_id,
            final_emotion=final_em,
            negative_score=round(smoothed_score, 4),
            dominant_modality=dominant_modality,
            is_negative_alert=is_alert,
        )

    def _cleanup_stale_tracks(self, active_ids: set[int]) -> None:
        for stale_tid in [tid for tid in self._score_history if tid not in active_ids]:
            del self._score_history[stale_tid]
        for stale_tid in [tid for tid in self._emotion_history if tid not in active_ids]:
            del self._emotion_history[stale_tid]