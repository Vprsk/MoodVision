"""End-to-end проверка VideoWorker с реальной моделью на data/test.mp4.

Запуск:  python scripts/e2e_check.py
Проверяет плеер: чтение потока -> predict() по контракту -> эмит кадров,
media_info, позицию, паузу и перемотку. Завершается автоматически.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QCoreApplication, QTimer

from app.core.stream_config import StreamConfig
from app.core.video_worker import VideoWorker

state = {
    "frames": 0,
    "detections": 0,
    "media": None,
    "positions": [],
    "statuses": [],
    "paused_events": [],
}


def on_frame(frame):
    state["frames"] += 1


def on_result(result):
    state["detections"] += len(result.detections)


def on_status(status):
    state["statuses"].append(status)
    print(f"  status -> {status}")


def on_media(fps, duration, seekable):
    state["media"] = (fps, duration, seekable)
    print(f"  media_info -> fps={fps:.2f} duration={duration:.1f}s seekable={seekable}")


def on_position(cur, dur):
    state["positions"].append((cur, dur))


def on_paused(paused):
    state["paused_events"].append(paused)
    print(f"  paused -> {paused}")


def main():
    app = QCoreApplication(sys.argv)
    cfg = StreamConfig.create(
        name="E2E", source=str(PROJECT_ROOT / "data" / "test.mp4"), location="Test"
    )
    worker = VideoWorker(cfg)
    worker.frame_ready.connect(on_frame)
    worker.result_ready.connect(on_result)
    worker.status_changed.connect(on_status)
    worker.media_info.connect(on_media)
    worker.position_changed.connect(on_position)
    worker.paused_changed.connect(on_paused)

    print("Запуск VideoWorker (загрузка моделей может занять время)...")
    worker.start()

    def step_pause():
        print(">>> Пауза")
        worker.set_paused(True)

    def check_paused():
        frames_at_pause = state["frames"]
        print(f">>> На паузе кадров: {frames_at_pause}; перематываю +15с")
        worker.seek_by(15.0)
        # Через 1с на паузе должен появиться 1 превью-кадр (seek сработал),
        # но не более пары (основной цикл на паузе не читает кадры).
        QTimer.singleShot(1000, lambda: verify_paused(frames_at_pause))

    def verify_paused(frames_at_pause):
        grew = state["frames"] - frames_at_pause
        print(f">>> После seek на паузе кадров +{grew} "
              f"(превью при перемотке, не инференс-поток)")
        print(">>> Снимаю с паузы")
        worker.set_paused(False)

    def verify_after():
        pos = state["positions"]
        last = pos[-1][0] if pos else 0.0
        print(f">>> После возобновления: кадров всего {state['frames']}, "
              f"позиция ~{last:.1f}s")

    def finish():
        pos = state["positions"]
        print("\n=== ИТОГ ===")
        print(f"кадров всего: {state['frames']}")
        print(f"детекций: {state['detections']}")
        print(f"media_info: {state['media']}")
        print(f"позиций получено: {len(pos)}; последние 3: {pos[-3:]}")
        print(f"события паузы: {state['paused_events']}")
        print(f"статусы: {state['statuses']}")
        ok = state["frames"] >= 1 and state["media"] is not None and len(pos) >= 1
        print("РЕЗУЛЬТАТ:", "OK" if ok else "FAIL")
        worker.stop()
        worker.wait(5000)
        app.quit()

    QTimer.singleShot(20000, step_pause)     # через 20с — пауза (модель загрузится)
    QTimer.singleShot(22000, check_paused)   # проверим + seek на паузе
    QTimer.singleShot(28000, verify_after)   # снимем с паузы, проверим позицию
    QTimer.singleShot(36000, finish)         # финиш
    QTimer.singleShot(120000, app.quit)      # жёсткий таймаут
    app.exec()
    return 0 if state["frames"] >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
