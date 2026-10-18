"""Видеоворкер: чтение потока + инференс модели в отдельном QThread.

Каждый видеопоток обслуживается собственным VideoWorker, который:
  * открывает источник через cv2.VideoCapture (RTSP / файл / веб-камера);
  * на КАЖДОМ кадре вызывает model.predict() контракта;
  * аннотирует кадр и отправляет его в GUI через сигнал frame_ready;
  * отправляет разбор результата (FrameResult) через сигнал result_ready;
  * переподключается с backoff при обрыве live-источника (RTSP/HTTP);
  * корректно освобождает ресурсы при остановке.

Важно: модель создаётся ВНУТРИ потока (в run()), т.к. YOLO/ViT содержат
состояние трекинга (persist=True) и не потокобезопасны. Поэтому каждый
поток получает свой независимый экземпляр пайплайна.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from app.core.annotate import draw_detection, draw_fps
from app.core.logging import get_logger
from app.core.stream_config import StreamConfig
from contracts.base import BaseFaceEmotionModel
from contracts.schemas import FrameResult

logger = get_logger("video_worker")

# Сколько последних кадров берём для оценки мгновенного FPS.
_FPS_WINDOW = 30

# Параметры переподключения live-источников после обрыва (см. _reconnect).
_RECONNECT_BASE_DELAY_S = 1.0    # пауза перед 2-й попыткой, дальше ×2
_RECONNECT_MAX_DELAY_S = 10.0    # потолок экспоненциального backoff
_RECONNECT_STEP_MS = 100         # шаг сна — чтобы быстро реагировать на stop()

# Таймауты сетевых источников (мс): не даём зависшему RTSP-соединению
# блокировать read()/stop() бесконечно (бэкенд FFmpeg, OpenCV ≥ 4.5).
_OPEN_TIMEOUT_MS = 5000
_READ_TIMEOUT_MS = 5000


class VideoWorker(QThread):
    """Рабочий поток для одного видеопотока с управлением воспроизведением.

    Помимо чтения и инференса, умеет:
      * приостанавливать/возобновлять показ (пауза);
      * перематыватьseek на ±N секунд (только для seekable-источников —
        видеофайлов; у RTSP/веб-камеры перемотка невозможна);
      * сообщать позицию, длительность и FPS источника для прогресс-бара;
      * переподключаться с экспоненциальным backoff при обрыве live-потока.

    Перемотка реализована через cv2.CAP_PROP_POS_FRAMES и применяется в
    рабочем потоке (в цикле run), т.к. трогать VideoCapture безопасно только
    из его собственного потока.
    """

    # Сигнал: аннотированный кадр (BGR numpy) готов к показу.
    frame_ready = Signal(np.ndarray)
    # Сигнал: результат обработки кадра (без самого кадра).
    result_ready = Signal(FrameResult)
    # Сигнал: текстовый статус потока (live / ошибка / остановлен и т.п.).
    status_changed = Signal(str)
    # Сигнал: мгновенный FPS обработки.
    fps_updated = Signal(float)
    # Сигнал: позиция воспроизведения (current_seconds, duration_seconds).
    position_changed = Signal(float, float)
    # Сигнал: изменилось состояние паузы.
    paused_changed = Signal(bool)
    # Сигнал: определены параметры источника (fps, длительность, seekable).
    media_info = Signal(float, float, bool)

    def __init__(self, config: StreamConfig, parent: Optional[object] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._running = False
        self._paused = False
        self._model: Optional[BaseFaceEmotionModel] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_times: Deque[float] = deque(maxlen=_FPS_WINDOW)

        # --- Параметры источника ---
        self._src_fps: float = 0.0
        self._duration: float = 0.0
        self._seekable: bool = config.is_seekable

        # Запрос перемотки в секундах (относительно текущей позиции).
        # Применяется в рабочем потоке, обнуляется после применения.
        self._pending_seek: Optional[float] = None

    @property
    def config(self) -> StreamConfig:
        return self._config

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def seekable(self) -> bool:
        return self._seekable

    # --- Управление воспроизведением (вызывается из GUI-потока) ------------

    def toggle_pause(self) -> None:
        """Переключает паузу. Для не-pause источников тоже допустимо."""
        self._paused = not self._paused
        self.paused_changed.emit(self._paused)

    def set_paused(self, paused: bool) -> None:
        """Явно задаёт состояние паузы."""
        if self._paused != paused:
            self._paused = paused
            self.paused_changed.emit(self._paused)

    def seek_by(self, seconds: float) -> None:
        """Просит перемотать на ±seconds (применится в рабочем потоке)."""
        if self._seekable:
            self._pending_seek = seconds

    def stop(self) -> None:
        """Просит поток остановиться (проверяется в цикле run)."""
        self._running = False
        # Снимаем паузу, чтобы цикл run не «залип» и увидел остановку.
        self._paused = False

    def _build_model(self) -> BaseFaceEmotionModel:
        """Создаёт экземпляр модели выбранного пайплайна.

        Импорт локальный, чтобы избежать цикличности и тяжёлых импортов
        на уровне модуля.
        """
        from app.core.pipeline_registry import create_pipeline

        return create_pipeline(
            self._config.pipeline_key, **self._config.pipeline_params
        )

    def _measure_fps(self) -> float:
        """Считает мгновенный FPS по окну последних кадров."""
        now = time.perf_counter()
        self._frame_times.append(now)
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    def _open_capture(self) -> cv2.VideoCapture:
        """Открывает источник видеопотока.

        Для индекса веб-камеры («0», «1», …) аргумент приводится к int:
        cv2.VideoCapture("0") со строкой интерпретирует «0» как имя файла
        и не открывает камеру. Для сетевых live-источников (RTSP/HTTP/RTMP)
        задаёт таймауты соединения и чтения: зависший поток иначе может
        навсегда заблокировать read(), из-за чего stop() не завершит воркер.
        Если бэкенд OpenCV не принял параметры — открываем без таймаутов
        (поведение как раньше).
        """
        source = self._config.capture_source
        # cv2 принимает индекс камеры только как int, не как строку.
        capture_arg: int | str = int(source) if source.isdigit() else source
        is_network = source.lower().startswith(
            ("rtsp://", "http://", "https://", "rtmp://")
        )
        open_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        read_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if is_network and open_prop is not None and read_prop is not None:
            cap = cv2.VideoCapture(
                capture_arg,
                cv2.CAP_ANY,
                [open_prop, _OPEN_TIMEOUT_MS, read_prop, _READ_TIMEOUT_MS],
            )
            if cap.isOpened():
                logger.info("Источник открыт: %s", source)
                return cap
            cap.release()
            logger.warning("Не удалось открыть сетевой источник с таймаутами: %s", source)
        cap = cv2.VideoCapture(capture_arg)
        if cap.isOpened():
            logger.info("Источник открыт: %s", source)
        else:
            logger.error("Не удалось открыть источник: %s", source)
        return cap

    def run(self) -> None:  # noqa: D102 (Qt-переопределение)
        self._running = True
        source = self._config.capture_source
        logger.info("Запуск воркера для потока %s (источник: %s)", self._config.stream_id, source)
        self.status_changed.emit("Подключение…")

        # --- Открываем источник ---
        self._cap = self._open_capture()
        if not self._cap.isOpened():
            self.status_changed.emit("Нет сигнала")
            logger.error("Нет сигнала от источника %s (поток %s)", source, self._config.stream_id)
            self._running = False
            return

        # --- Читаем метаданные источника (fps, длительность) ---
        self._read_media_meta()

        # --- Создаём модель ВНУТРИ потока ---
        try:
            self._model = self._build_model()
        except Exception as exc:  # noqa: BLE001 — показываем любую ошибку загрузки
            self.status_changed.emit(f"Ошибка модели: {exc}")
            logger.error("Ошибка загрузки модели для потока %s: %s", self._config.stream_id, exc)
            self._release()
            return

        self.status_changed.emit("В эфире")
        logger.info("Поток %s в эфире (модель загружена)", self._config.stream_id)
        frame_id = 0

        try:
            while self._running:
                # --- Пауза: кадры не обрабатываем, но реагируем на перемотку
                # и возобновление. При перемотке на паузе показываем одиночный
                # кадр на новой позиции (без инференса) для отзывчивости. ---
                if self._paused:
                    if self._pending_seek is not None:
                        if self._apply_pending_seek():
                            self._emit_preview_frame()
                    self.msleep(30)
                    continue

                # --- Применяем отложенную перемотку (если просили) ---
                if self._pending_seek is not None:
                    self._apply_pending_seek()

                ret, frame = self._cap.read()
                if not ret:
                    # Для файлов это конец записи. Для live-источника
                    # (RTSP/HTTP/веб-камера) длительность неизвестна, поэтому
                    # такой EOF — почти всегда сетевой обрыв, а не конец.
                    if self._config.is_live and self._duration <= 0.0:
                        logger.warning(
                            "Потерян сигнал от live-источника %s (поток %s), пробуем переподключение…",
                            source,
                            self._config.stream_id,
                        )
                        self.status_changed.emit("Переподключение…")
                        if self._reconnect():
                            continue
                        logger.error(
                            "Переподключение к %s не удалось, поток %s остановлен",
                            source,
                            self._config.stream_id,
                        )
                        break  # stop() во время переподключения
                    self.status_changed.emit("Поток завершён")
                    logger.info("Поток %s завершён (конец файла)", self._config.stream_id)
                    break

                frame_id += 1
                loop_start = time.perf_counter()

                result: FrameResult = self._model.predict(frame_id, frame)

                # Аннотируем копию кадра для отображения.
                annotated = frame.copy()
                for det in result.detections:
                    draw_detection(annotated, det)

                fps = self._measure_fps()
                draw_fps(annotated, fps)

                # Отдаём кадр в GUI. annotated уже копия frame (строка выше),
                # дополнительного copy() не требуется.
                self.frame_ready.emit(annotated)
                self.result_ready.emit(result)
                self.fps_updated.emit(fps)

                # Сообщаем позицию для прогресс-бара.
                current = self._current_position_seconds()
                self.position_changed.emit(current, self._duration)

                # Небольшая пауза, чтобы не грузить CPU на 100% при высоком FPS.
                elapsed = time.perf_counter() - loop_start
                if elapsed < 0.001:
                    self.msleep(1)
        finally:
            self._release()
            logger.info("Воркер потока %s завершён", self._config.stream_id)

    def _read_media_meta(self) -> None:
        """Читает FPS и длительность источника, эмитит media_info."""
        assert self._cap is not None
        self._src_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = float(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if self._src_fps > 0.0 and total_frames > 0.0:
            self._duration = total_frames / self._src_fps
        else:
            # Для live-источников длительность неизвестна.
            self._duration = 0.0
            self._seekable = False
        self.media_info.emit(self._src_fps, self._duration, self._seekable)

    def _current_position_seconds(self) -> float:
        """Текущая позиция воспроизведения в секундах."""
        if self._cap is None:
            return 0.0
        pos_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        return pos_ms / 1000.0

    def _apply_pending_seek(self) -> bool:
        """Применяет отложенный запрос перемотки (±секунды).

        Вызывается ТОЛЬКО из рабочего потока. Возвращает True, если перемотка
        реально выполнена. Для неперематываемых источников — False.
        """
        if self._pending_seek is None:
            return False
        if not self._seekable or self._cap is None:
            self._pending_seek = None
            return False
        delta = self._pending_seek
        self._pending_seek = None

        fps = self._src_fps if self._src_fps > 0.0 else 25.0
        frame_delta = int(round(delta * fps))
        if frame_delta == 0:
            return False
        current_frame = float(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)
        total_frames = float(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        target = current_frame + frame_delta
        if target < 0:
            target = 0
        if total_frames > 0 and target >= total_frames:
            target = max(0.0, total_frames - 1)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        return True

    def _interruptible_sleep(self, seconds: float) -> bool:
        """Спит мелкими шагами, чтобы быстро отреагировать на stop().

        Возвращает False, если поток был остановлен до конца паузы.
        """
        total_ms = int(seconds * 1000)
        slept_ms = 0
        while slept_ms < total_ms and self._running:
            step = min(_RECONNECT_STEP_MS, total_ms - slept_ms)
            self.msleep(step)
            slept_ms += step
        return self._running

    def _reconnect(self) -> bool:
        """Переподключает live-источник после обрыва чтения.

        Backoff: первая попытка сразу, затем 1 c → 2 c → 4 c → 8 c → … ≤ 10 c.
        Число попыток не ограничено, пока не пришёл stop() — кратковременный
        обрыв сети не должен убивать камеру навсегда (до ручного перезапуска).

        Возвращает True, если источник снова открыт и цикл run продолжается;
        False — поток остановлен во время переподключения.
        """
        self.status_changed.emit("Обрыв потока — переподключение…")
        attempt = 0
        while self._running:
            attempt += 1
            if attempt > 1:
                delay = min(
                    _RECONNECT_BASE_DELAY_S * (2 ** (attempt - 2)),
                    _RECONNECT_MAX_DELAY_S,
                )
                self.status_changed.emit(f"Переподключение… (попытка {attempt})")
                if not self._interruptible_sleep(delay):
                    break
            if self._cap is not None:
                self._cap.release()
            self._cap = self._open_capture()
            if self._cap.isOpened():
                # FPS-окно накапливалось до обрыва — сбрасываем метрику.
                self._frame_times.clear()
                self._read_media_meta()
                self.status_changed.emit("В эфире")
                return True
        return False

    def _emit_preview_frame(self) -> None:
        """Читает и показывает один кадр без инференса (для паузы+перемотка)."""
        if self._cap is None:
            return
        ret, frame = self._cap.read()
        if not ret:
            return
        self.frame_ready.emit(frame.copy())
        self.position_changed.emit(
            self._current_position_seconds(), self._duration
        )

    def _release(self) -> None:
        """Освобождает ресурсы захвата. Вызывается в finally."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
