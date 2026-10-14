from collections.abc import Iterator

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from cam_reader import CameraReader
from contracts.schemas import FrameResult
from mood_vision import MoodVision


class Pipeline:
    """Объединяет захват кадров и модель распознавания лиц/эмоций."""

    def __init__(
        self,
        camera: CameraReader,
        model: MoodVision,
    ) -> None:
        self.cam: CameraReader = camera
        self.model: MoodVision = model

    def process_frames(
        self,
    ) -> Iterator[tuple[NDArray[np.uint8], FrameResult]]:
        """Читает кадры и возвращает пары (кадр, результат инференса)."""
        for frame_id, frame in self.cam.get_frames():
            result = self.model.predict(
                frame_id=frame_id,
                frame=frame,
            )

            yield frame, result

    def run(self) -> None:
        """Запускает обработку видеопотока и отображает результат."""
        window_name = "Face emotion pipeline"

        try:
            for frame, result in self.process_frames():
                self._draw_result(frame, result)
                cv.imshow(window_name, frame)

                if cv.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        finally:
            cv.destroyAllWindows()

    def _draw_result(
        self,
        frame: NDArray[np.uint8],
        result: FrameResult,
    ) -> None:
        """Рисует рамки, ID трека и эмоции на кадре."""
        for face in result.detections:
            x1 = face.bbox.x_min
            y1 = face.bbox.y_min
            x2 = face.bbox.x_max
            y2 = face.bbox.y_max

            color = (0, 255, 0)

            cv.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            track_id = face.track_id if face.track_id is not None else "—"
            label = f"ID: {track_id} | {face.emotion} {face.emotion_confidence:.2f}"

            label_y = max(y1 - 10, 25)

            cv.putText(
                frame,
                label,
                (x1, label_y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv.LINE_AA,
            )


if __name__ == "__main__":
    camera = CameraReader(source="test.mp4")

    model = MoodVision(
        face_model_path="checkpoints/yolov8n-face.pt",
        emotion_model_path="trpakov/vit-face-expression",
        face_conf=0.3,
        emotion_conf=0.5,
        device=0,
        tracker="bytetrack.yaml",
    )

    pipeline = Pipeline(camera, model)
    pipeline.run()