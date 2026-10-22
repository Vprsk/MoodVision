"""Песочница try.py с поддержкой реального аудио из видеофайла и ручным тестом звука.

Управление в окне OpenCV:
  [A] — подать тестовый громкий звук / крик (проверка реакции пайплайна на аудио)
  [M] — включить / выключить звук (Mute)
  [1] — переключить на Сцену 1 (Лицо + Звук)
  [2] — переключить на Сцену 2 (Лицо + Звук + Тело)
  [3] — переключить на Сцену 3 (Звук + Тело)
  [ESC] — выход
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Поднимаемся на 3 уровня вверх от mood_base/scripts/ до корня проекта (MoodVision)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contracts import BaseFaceEmotionModel, Emotion, FaceDetection, FrameResult
from mood_base.models import (
    Scene1FaceAudioModel,
    Scene2FaceBodyAudioModel,
    Scene3BodyAudioModel,
)

VIDEO_SOURCE = str(PROJECT_ROOT / "data" / "test.mp4")
SAMPLE_RATE = 16000
AUDIO_WINDOW_SEC = 1.5  # Окно аудио 1.5 секунды

EMOTION_COLORS = {
    Emotion.UNKNOWN: (0, 165, 255),
    Emotion.HAPPY: (0, 255, 255),
    Emotion.SAD: (255, 0, 0),
    Emotion.ANGRY: (0, 0, 255),
    Emotion.FEAR: (255, 0, 255),
    Emotion.DISGUST: (128, 0, 128),
    Emotion.SURPRISE: (0, 255, 127),
    Emotion.NEUTRAL: (0, 255, 0),
}
DEFAULT_COLOR = (0, 255, 0)


def load_full_audio_track(video_path: str, sample_rate: int = 16000) -> np.ndarray | None:
    """Извлекает аудиодорожку из видеофайла в моно float32 [-1, 1] (16 кГц)."""
    # Попытка 1: через torchaudio (уже есть в requirements.txt)
    try:
        import torchaudio

        waveform, sr = torchaudio.load(video_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        audio_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
        print(f"[AUDIO] Аудиодорожка загружена через torchaudio ({len(audio_np) / sample_rate:.1f} сек)")
        return audio_np
    except Exception as exc:
        print(f"[AUDIO] torchaudio не смог открыть аудио напрямую ({exc}), пробуем ffmpeg...")

    # Попытка 2: через системный ffmpeg (если установлен в PATH)
    if shutil.which("ffmpeg"):
        try:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-i",
                video_path,
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-loglevel",
                "quiet",
                "-",
            ]
            raw_bytes = subprocess.check_output(cmd)
            audio_np = np.frombuffer(raw_bytes, dtype=np.float32)
            if audio_np.size > 0:
                print(f"[AUDIO] Аудиодорожка загружена через ffmpeg ({len(audio_np) / sample_rate:.1f} сек)")
                return audio_np
        except Exception as exc:
            print(f"[AUDIO] В видео нет аудиодорожки или ошибка ffmpeg: {exc}")

    print(
        "[AUDIO] Реальная аудиодорожка не найдена. "
        "Используй клавишу [A] в окне видео для подачи тестового звука/крика!"
    )
    return None


def make_synthetic_angry_voice(sample_rate: int = 16000, duration: float = 1.5) -> np.ndarray:
    """Генерирует громкий модулированный сигнал (имитация крика/возмущённого голоса + пик)."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Смесь речевых гармоник (180 Гц + 360 Гц + 720 Гц) с амплитудной модуляцией и шумом
    carrier = (
        0.45 * np.sin(2 * np.pi * 185 * t)
        + 0.30 * np.sin(2 * np.pi * 370 * t)
        + 0.15 * np.sin(2 * np.pi * 740 * t)
    )
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 6.0 * t)
    noise = np.random.normal(0, 0.08, size=t.shape).astype(np.float32)
    signal = np.clip((carrier * envelope) + noise, -0.95, 0.95).astype(np.float32)
    return signal


def draw_result(frame: np.ndarray, det: FaceDetection, extra_text: str = "") -> None:
    """Отрисовывает рамку детекции и статус."""
    x1, y1 = det.bbox.x_min, det.bbox.y_min
    x2, y2 = det.bbox.x_max, det.bbox.y_max

    color = EMOTION_COLORS.get(det.emotion, DEFAULT_COLOR)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID {det.track_id} | {det.emotion.value} {det.emotion_confidence:.2f}{extra_text}"
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_audio_hud(frame: np.ndarray, result: FrameResult, muted: bool, spike_frames_left: int) -> None:
    """Рисует панель состояния звука в верхней части экрана."""
    audio = result.audio
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (10, 45), (min(w - 10, 620), 115), (25, 25, 25), -1)

    if muted:
        status_text = "AUDIO: MUTED (Press [M] to unmute)"
        status_color = (128, 128, 128)
    elif spike_frames_left > 0:
        status_text = f"AUDIO TEST SPIKE ACTIVE! ({spike_frames_left} frames left)"
        status_color = (0, 0, 255)
    elif audio is not None and (audio.speech_detected or audio.is_loud_spike):
        status_text = (
            f"AUDIO ACTIVE | Emotion: {audio.audio_emotion.value.upper()} "
            f"({audio.audio_confidence:.2f}) | RMS: {audio.rms_energy:.3f} "
            f"| Spike: {audio.is_loud_spike}"
        )
        status_color = (0, 255, 255) if not audio.is_loud_spike else (0, 0, 255)
    else:
        rms_val = audio.rms_energy if audio is not None else 0.0
        status_text = f"AUDIO SILENCE | RMS: {rms_val:.4f} (Press [A] to inject loud voice/scream)"
        status_color = (180, 180, 180)

    cv2.putText(
        frame,
        status_text,
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        status_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Keys: [A]=Inject Scream/Voice  [M]=Mute  [1/2/3]=Switch Scene  [ESC]=Exit",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


class VideoApplication:
    """Тестовый стенд для проверки мультимодальных сцен со звуком."""

    def __init__(self, model: BaseFaceEmotionModel):
        self.model = model

    def process_video(self, video_source: str) -> None:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник: {video_source}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        audio_track = load_full_audio_track(video_source, SAMPLE_RATE)
        window_samples = int(AUDIO_WINDOW_SEC * SAMPLE_RATE)

        frame_id = 0
        muted = False
        inject_spike_frames = 0  # Сколько кадров подавать искусственный крик после нажатия [A]

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Видеопоток завершён")
                    break

                frame_id += 1
                loop_start = time.perf_counter()

                # 1. Формируем синхронный аудио-чанк для текущего кадра
                audio_chunk: np.ndarray | None = None
                if not muted:
                    if inject_spike_frames > 0:
                        audio_chunk = make_synthetic_angry_voice(SAMPLE_RATE, AUDIO_WINDOW_SEC)
                        inject_spike_frames -= 1
                    elif audio_track is not None:
                        end_sample = int((frame_id / video_fps) * SAMPLE_RATE)
                        start_sample = max(0, end_sample - window_samples)
                        if end_sample > start_sample and start_sample < len(audio_track):
                            audio_chunk = audio_track[start_sample : min(end_sample, len(audio_track))]

                # 2. Вызываем пайплайн с передачей видеокадра И аудиочанка
                result: FrameResult = self.model.predict(
                    frame_id, frame, audio_chunk=audio_chunk
                )

                # Карта агрегированных событий по track_id
                events_map = {ev.track_id: ev for ev in result.track_events}

                # 3. Отрисовка и вывод в консоль
                for det in result.detections:
                    ev = events_map.get(det.track_id) if det.track_id is not None else None
                    extra = ""
                    if ev is not None:
                        alert_flag = " [ALERT!]" if ev.is_negative_alert else ""
                        extra = f" | neg:{ev.negative_score:.2f} ({ev.dominant_modality}){alert_flag}"
                        # Чтобы при срабатывании аудио-алерта менялся цвет и текст рамки в UI:
                        if ev.dominant_modality == "audio" or ev.is_negative_alert:
                            det = det.model_copy(
                                update={
                                    "emotion": ev.final_emotion,
                                    "emotion_confidence": max(det.emotion_confidence, ev.negative_score),
                                }
                            )

                    draw_result(frame, det, extra_text=extra)

                    print(
                        {
                            "frame_id": result.frame_id,
                            "track_id": det.track_id,
                            "emotion": det.emotion.value,
                            "neg_score": ev.negative_score if ev else 0.0,
                            "dom_mod": ev.dominant_modality if ev else "none",
                            "alert": ev.is_negative_alert if ev else False,
                            "audio_em": result.audio.audio_emotion.value if result.audio else "none",
                            "audio_rms": result.audio.rms_energy if result.audio else 0.0,
                        }
                    )

                fps = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | Model: {self.model.__class__.__name__}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                draw_audio_hud(frame, result, muted, inject_spike_frames)

                cv2.imshow("Multimodal Affect Recognition (Audio + Video Test)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key in (ord("a"), ord("A")):
                    # Подаём 20 кадров (~1-2 секунды) громкого звука/крика для проверки
                    inject_spike_frames = 20
                    print("\n>>> [TEST] Инжекция громкого аудио (20 кадров)! Смотри на dom_mod и neg_score <<<\n")
                elif key in (ord("m"), ord("M")):
                    muted = not muted
                    print(f"\n>>> [TEST] Звук {'ВЫКЛЮЧЕН (Mute)' if muted else 'ВКЛЮЧЕН'} <<<\n")
                elif key == ord("1"):
                    print("\n>>> Переключение на Сцену 1 (Лицо + Звук) <<<\n")
                    self.model = Scene1FaceAudioModel()
                elif key == ord("2"):
                    print("\n>>> Переключение на Сцену 2 (Лицо + Звук + Тело) <<<\n")
                    self.model = Scene2FaceBodyAudioModel()
                elif key == ord("3"):
                    print("\n>>> Переключение на Сцену 3 (Звук + Тело) <<<\n")
                    self.model = Scene3BodyAudioModel()
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    model = Scene2FaceBodyAudioModel()
    app = VideoApplication(model=model)
    app.process_video(VIDEO_SOURCE)


if __name__ == "__main__":
    main()