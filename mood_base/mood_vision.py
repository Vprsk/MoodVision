import numpy as np
from numpy.typing import NDArray

from contracts.base import BaseFaceEmotionModel
from contracts.schemas import FaceDetection, FrameResult

from mood_base.face_detect import YoloFaceDetector
from mood_base.mood_class import HuggingFaceEmotionClassifier


class MoodVision(BaseFaceEmotionModel):
    """Пайплайн: детекция лиц YOLO -> классификация эмоций"""

    def __init__(
        self,
        face_model_path: str = "mood_base/model/yolov8n-face.pt",
        emotion_model_path: str = "trpakov/vit-face-expression",
        face_conf: float = 0.3,
        emotion_conf: float = 0.5,
        yolo_device: str | int = 0,
        hf_device: str | int = 0,
        tracker: str = "trackers/tracktrack.yaml",
        face_verbose: bool = False,
    ) -> None:
        self.face_detector = YoloFaceDetector(
            model_path=face_model_path,
            conf=face_conf,
            device=yolo_device,
            tracker=tracker,
            verbose=face_verbose
        )

        self.emotion_classifier = HuggingFaceEmotionClassifier(
            model_path=emotion_model_path,
            device=hf_device,
        )

    def predict(
        self,
        frame_id: int,
        frame: NDArray[np.uint8],
    ) -> FrameResult:
        """Детектирует лица, определяет эмоции и формирует FrameResult"""
        faces = self._detect_faces(frame)
        faces = self._classify_emotions(frame, faces)

        return self._build_result(
            frame_id=frame_id,
            faces=faces,
        )

    def _detect_faces(self, frame: NDArray[np.uint8]) -> list[FaceDetection]:
        """Находит лица в кадре"""
        return self.face_detector.detect(frame)

    def _classify_emotions(
        self,
        frame: NDArray[np.uint8],
        faces: list[FaceDetection],
    ) -> list[FaceDetection]:
        """Определяет эмоцию для каждого найденного лица"""
        classified_faces: list[FaceDetection] = []

        for face in faces:
            try:
                classified_face = self.emotion_classifier.predict(
                    frame=frame,
                    face=face,
                )
            except ValueError:
                classified_face = face

            classified_faces.append(classified_face)

        return classified_faces

    def _build_result(
        self,
        frame_id: int,
        faces: list[FaceDetection],
    ) -> FrameResult:
        """Собирает итоговый FrameResult."""
        return FrameResult(
            frame_id=frame_id,
            detections=faces,
        )
