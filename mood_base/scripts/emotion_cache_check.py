"""Проверка кэша модели эмоций: ViT не перечитывается с HF повторно.

Запуск:  env/Scripts/python scripts/emotion_cache_check.py

Проверяет:
  * пайплайн эмоций строится БЕЗ обращения к сети (локальный кэш HF);
  * два экземпляра CurrentYoloVitModel делят ОДИН объект классификатора;
  * predict() работает на общем классификаторе.
"""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from mood_base.models import CurrentYoloVitModel
from mood_base.models.current_yolo_vit import (
    EMOTION_MODEL_ID,
    _EMOTION_PIPELINE_CACHE,
    _get_emotion_pipeline,
)


def main() -> int:
    # --- Выбор устройства — как в CurrentYoloVitModel ---
    if torch.cuda.is_available():
        device, dtype = 0, torch.float16
    else:
        device, dtype = -1, torch.float32

    # --- 1) Загрузка без сети: рвём connect, HF обязан обойтись кэшем ---
    original_connect = socket.socket.connect

    def _blocked(self, *args, **kwargs):  # noqa: ANN001, ANN202
        raise RuntimeError("сеть отключена: модель должна быть в локальном кэше HF")

    socket.socket.connect = _blocked  # type: ignore[method-assign]
    try:
        t0 = time.perf_counter()
        clf = _get_emotion_pipeline(EMOTION_MODEL_ID, device, dtype)
        t1 = time.perf_counter()
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
    assert clf is not None
    print(f"OK: пайплайн эмоций собран офлайн за {t1 - t0:.2f} с")

    # --- 2) Повторный вызов возвращает тот же объект (кэш процесса) ---
    t0 = time.perf_counter()
    clf_again = _get_emotion_pipeline(EMOTION_MODEL_ID, device, dtype)
    t1 = time.perf_counter()
    assert clf_again is clf, "кэш не сработал: пайплайн пересоздан"
    print(f"OK: повторный запрос отдал кэш за {t1 - t0:.4f} с")

    # --- 3) Два экземпляра модели делят один классификатор ---
    t0 = time.perf_counter()
    first = CurrentYoloVitModel()
    t1 = time.perf_counter()
    second = CurrentYoloVitModel()
    t2 = time.perf_counter()
    assert first.emotion_classifier is clf, "модель не взяла пайплайн из кэша"
    assert (
        second.emotion_classifier is first.emotion_classifier
    ), "классификатор не переиспользован между экземплярами"
    print(
        f"OK: общий классификатор у двух экземпляров "
        f"(создание: {t1 - t0:.1f} с → {t2 - t1:.1f} с)"
    )
    assert len(_EMOTION_PIPELINE_CACHE) >= 1, "кэш пуст"

    # --- 4) Инференс на общем классификаторе ---
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    r1 = first.predict(0, frame)
    r2 = second.predict(1, frame)
    print(f"OK: predict() работает (detections: {len(r1.detections)}/{len(r2.detections)})")

    print("OK: кэш модели эмоций работает")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
