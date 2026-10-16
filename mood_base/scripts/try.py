"""Тонкий слой приложения (экс-прототип try.py).

Знает ТОЛЬКО про контракт BaseFaceEmotionModel: читает видео, вызывает
predict() и рисует результат. Конкретная модель подставляется через DI.
"""

import sys
import time
from pathlib import Path

import cv2

# Позволяет запускать `python scripts/try.py` из корня проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contracts import BaseFaceEmotionModel, Emotion, FaceDetection, FrameResult
from mood_base.models import CurrentYoloVitModel

VIDEO_SOURCE = "data/test.mp4"

# Цвет рамки по эмоции (default — зелёный)
EMOTION_COLORS = {
    Emotion.UNKNOWN: (0, 165, 255),
    Emotion.HAPPY: (0, 255, 255),
    Emotion.SAD: (255, 0, 0),
    Emotion.ANGRY: (0, 0, 255),
}
DEFAULT_COLOR = (0, 255, 0)


def draw_result(frame, det: FaceDetection) -> None:
    """Отрисовывает одну детекцию. Уровень приложения: ничего не знает о моделях."""
    x1, y1 = det.bbox.x_min, det.bbox.y_min
    x2, y2 = det.bbox.x_max, det.bbox.y_max

    color = EMOTION_COLORS.get(det.emotion, DEFAULT_COLOR)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID {det.track_id} | {det.emotion.value} {det.emotion_confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, max(15, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


class VideoApplication:
    """Приложение работает только со стандартом (FrameResult), а не с моделями."""

    def __init__(self, model: BaseFaceEmotionModel):
        self.model = model

    def process_video(self, video_source: str) -> None:
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник: {video_source}")

        frame_id = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Видеопоток завершён или кадр не получен")
                    break

                frame_id += 1
                loop_start = time.perf_counter()

                # Стандартный вызов модели через контракт
                result: FrameResult = self.model.predict(frame_id, frame)

                # Приложение работает со стандартом
                for det in result.detections:
                    draw_result(frame, det)
                    print(
                        {
                            "frame_id": result.frame_id,
                            "track_id": det.track_id,
                            "face_conf": round(det.face_confidence, 3),
                            "bbox": [
                                det.bbox.x_min,
                                det.bbox.y_min,
                                det.bbox.x_max,
                                det.bbox.y_max,
                            ],
                            "emotion": det.emotion.value,
                            "emotion_conf": round(det.emotion_confidence, 3),
                        }
                    )

                fps = 1.0 / max(time.perf_counter() - loop_start, 1e-6)

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Face emotion recognition (CUDA)", frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    # DI: сюда можно подставить любую реализацию контракта
    model = CurrentYoloVitModel()
    app = VideoApplication(model=model)
    app.process_video(VIDEO_SOURCE)


if __name__ == "__main__":
    main()
