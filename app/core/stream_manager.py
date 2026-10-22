"""Менеджер видеопотоков: владеет воркерами, агрегирует состояние треков.

Отвечает за:
  * создание/запуск/остановку VideoWorker (по одному QThread на поток);
  * агрегацию наблюдений эмоций в ClientTrack по track_id;
  * рассылку сигналов для обновления UI.

Кадры (numpy) идут напрямую от воркера к виджету через frame_ready — этот
класс управляет жизненным циклом потоков и агрегирует данные детекций.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from PySide6.QtCore import QObject, Signal

from app.core.logging import get_logger
from app.core.stream_config import (
    STATUS_INCIDENT,
    STATUS_NORMAL,
    STATUS_UNSURE,
    ClientTrack,
    StreamConfig,
)
from app.core.video_worker import VideoWorker
from contracts.schemas import FrameResult

logger = get_logger("stream_manager")

# Сколько секунд трек не появляется в кадре, чтобы считаться «ушёдшим».
TRACK_TTL_SECONDS = 10.0

# Сколько мс ждать завершения воркера в stop_stream. Если инференс кадра
# дольше — воркер остаётся в _stopping_workers до фактического finished.
STOP_WAIT_MS = 3000

# Сколько мс ждать завершения всех воркеров при закрытии приложения. Здесь
# ждать можно дольше: первая загрузка модели (YOLO + ViT) занимает секунды,
# и без этого ожидания Qt ругался бы «QThread: Destroyed while thread is
# still running».
SHUTDOWN_WAIT_MS = 20000

# Сколько мс ждать завершения воркера при удалении потока. Удаление вызывается
# из GUI-потока, поэтому ждать долго нельзя — интерфейс «залипнет». Если воркер
# не успел завершить текущий кадр, он доживёт сам: ссылка на него остаётся в
# _stopping_workers до сигнала finished.
REMOVE_WAIT_MS = 400


@dataclass
class StreamRuntime:
    """Среднее время выполнения одного потока: конфиг + воркер + треки."""

    config: StreamConfig
    worker: Optional[VideoWorker] = None
    tracks: Dict[int, ClientTrack] = field(default_factory=dict)
    last_status: str = "Выключено"
    fps: float = 0.0
    # True, когда stop() уже запрошен, но поток ещё не завершился:
    # воркер жив (ссылка сохранена), но для UI поток уже «выключен».
    stopping: bool = False

    @property
    def is_live(self) -> bool:
        return (
            self.worker is not None
            and self.worker.isRunning()
            and not self.stopping
        )


class StreamManager(QObject):
    """Управляет всеми видеопотоками приложения."""

    # Сигналы для UI: обновление конкретного потока.
    stream_status_changed = Signal(str, str)      # stream_id, status
    stream_fps_changed = Signal(str, float)       # stream_id, fps
    tracks_updated = Signal(str, list)            # stream_id, list[ClientTrack]
    incident_raised = Signal(str, str)            # stream_id, описание
    stream_finished = Signal(str)                 # stream_id (поток завершился)
    stream_removed = Signal(str)                  # stream_id (поток удалён)
    # Миниатюра потока: stream_id + уменьшенный кадр (BGR numpy).
    frame_preview = Signal(str, object)           # stream_id, frame

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._runtimes: Dict[str, StreamRuntime] = {}
        # Воркеры с отправленным stop(), но ещё не завершившиеся (wait()
        # истёк). Сильные ссылки не дают QThread уничтожиться «в работе».
        self._stopping_workers: Set[VideoWorker] = set()

    # --- Управление потоками -------------------------------------------------

    def add_stream(self, config: StreamConfig) -> StreamRuntime:
        """Регистрирует конфигурацию потока (без запуска)."""
        runtime = StreamRuntime(config=config)
        self._runtimes[config.stream_id] = runtime
        return runtime

    def start_stream(self, stream_id: str) -> bool:
        """Создаёт и запускает VideoWorker для указанного потока."""
        runtime = self._runtimes.get(stream_id)
        if runtime is None:
            return False
        if runtime.is_live:
            return True

        # Старый воркер мог ещё завершаться (stop с таймаутом wait()) —
        # сохраняем ссылку до его finished, иначе QThread умрёт «в работе».
        previous = runtime.worker
        if previous is not None and previous.isRunning():
            self._stopping_workers.add(previous)

        worker = VideoWorker(runtime.config)
        worker.frame_ready.connect(self._on_frame_ready)
        worker.preview_ready.connect(
            lambda frame, rid=stream_id: self._on_preview_ready(rid, frame)
        )
        worker.result_ready.connect(
            lambda result, rid=stream_id: self._on_result(rid, result)
        )
        worker.status_changed.connect(
            lambda status, rid=stream_id: self._on_status(rid, status)
        )
        worker.fps_updated.connect(
            lambda fps, rid=stream_id: self._on_fps(rid, fps)
        )
        worker.finished.connect(lambda rid=stream_id: self._on_finished(rid))
        worker.start()
        runtime.worker = worker
        runtime.stopping = False
        logger.info("Поток %s запущен (источник: %s)", stream_id, runtime.config.capture_source)
        return True

    def stop_stream(self, stream_id: str, wait_ms: int = STOP_WAIT_MS) -> None:
        """Останавливает воркер потока (мягко просит завершиться).

        wait_ms — сколько мс ждать фактического завершения воркера. Если
        воркер не успел (долгий инференс кадра), он доживает сам: ссылка
        удерживается в _stopping_workers до сигнала finished.
        """
        runtime = self._runtimes.get(stream_id)
        if runtime is None or runtime.worker is None:
            return
        worker = runtime.worker
        runtime.stopping = True
        worker.stop()
        if worker.wait(wait_ms):
            # Поток завершился за отведённое время — ссылку можно снимать:
            # уничтожение завершённого QThread безопасно.
            self._stopping_workers.discard(worker)
            if runtime.worker is worker:
                runtime.worker = None
            runtime.stopping = False
            logger.info("Поток %s остановлен", stream_id)
        else:
            # Таймаут: инференс кадра ещё выполняется. НЕ обнуляем ссылку —
            # иначе QThread может быть уничтожен во время работы («QThread:
            # Destroyed while thread is still running»). Сильную ссылку
            # держим до фактического finished (снимет _on_finished);
            # в _stopping_workers воркер переживает даже remove_stream/stop_all.
            self._stopping_workers.add(worker)
            logger.warning(
                "Поток %s не успел остановиться за %s мс, вордер удерживается до finished",
                stream_id,
                wait_ms,
            )
        runtime.last_status = "Выключено"
        runtime.fps = 0.0
        self.stream_status_changed.emit(stream_id, runtime.last_status)

    def remove_stream(self, stream_id: str) -> bool:
        """Останавливает и удаляет поток из менеджера.

        Возвращает True, если поток был найден и удалён. Ожидание завершения
        воркера намеренно короткое (REMOVE_WAIT_MS): метод вызывается из
        GUI-потока, и долгое ожидание «подвешивало» бы интерфейс. Если воркер
        не успел завершить кадр, он доживёт сам — ссылка на него остаётся в
        _stopping_workers до сигнала finished.
        """
        runtime = self._runtimes.get(stream_id)
        if runtime is None:
            return False

        worker = runtime.worker
        if worker is not None:
            runtime.stopping = True
            worker.stop()
            if worker.wait(REMOVE_WAIT_MS):
                # Воркер завершился — ссылку можно снять безопасно.
                self._stopping_workers.discard(worker)
                runtime.worker = None
                runtime.stopping = False
            else:
                # Воркер ещё жив: удерживаем ссылку до finished, иначе QThread
                # будет уничтожен во время работы (аварийное завершение).
                self._stopping_workers.add(worker)
                logger.warning(
                    "Поток %s удалён, но воркер ещё работает — удерживается до finished",
                    stream_id,
                )

        # Убираем runtime из реестра: поток больше не отображается в UI и
        # не участвует в сохранении состояния.
        self._runtimes.pop(stream_id, None)
        self.stream_removed.emit(stream_id)
        logger.info("Поток %s удалён из менеджера", stream_id)
        return True

    def stop_all(self) -> None:
        """Останавливает все потоки (при закрытии приложения)."""
        for stream_id in list(self._runtimes.keys()):
            self.stop_stream(stream_id)
        # Страховка на выход: воркеры, не уложившиеся в таймаут stop_stream
        # (например, ещё грузится модель), ещё немного ждём, чтобы приложение
        # не завершилось с живыми QThread.
        pending = [w for w in self._stopping_workers if w.isRunning()]
        if not pending:
            return
        deadline = time.monotonic() + SHUTDOWN_WAIT_MS / 1000.0
        for w in pending:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                break
            w.wait(remaining_ms)
        still_running = [w for w in self._stopping_workers if w.isRunning()]
        if still_running:
            logger.warning(
                "Не все воркеры завершились при закрытии: %d", len(still_running)
            )

    def toggle_stream(self, stream_id: str) -> None:
        """Переключает состояние потока: запуск <-> остановка."""
        runtime = self._runtimes.get(stream_id)
        if runtime is None:
            return
        if runtime.is_live:
            self.stop_stream(stream_id)
        else:
            self.start_stream(stream_id)

    # --- Запросы состояния ---------------------------------------------------

    def get_runtime(self, stream_id: str) -> Optional[StreamRuntime]:
        return self._runtimes.get(stream_id)

    def all_runtimes(self) -> List[StreamRuntime]:
        return list(self._runtimes.values())

    def locations(self) -> List[str]:
        """Уникальные локации потоков (для группировки в UI)."""
        seen: dict[str, None] = {}
        for rt in self._runtimes.values():
            seen.setdefault(rt.config.location or "Без группы", None)
        return list(seen.keys())


    # --- Обработчики сигналов воркеров --------------------------------------

    def _on_frame_ready(self, frame) -> None:  # noqa: ANN001
        # Кадры обрабатываются напрямую в UI через собственные подключения,
        # слот оставлен для будущей централизованной обработки/записи.
        return

    def _on_preview_ready(self, stream_id: str, frame) -> None:  # noqa: ANN001
        """Пересылает кадр-превью в UI (миниатюра в списке потоков)."""
        runtime = self._runtimes.get(stream_id)
        if runtime is None or runtime.stopping:
            return
        self.frame_preview.emit(stream_id, frame)

    def _on_status(self, stream_id: str, status: str) -> None:
        runtime = self._runtimes.get(stream_id)
        if runtime is None or runtime.stopping:
            # Пока идёт остановка, поздние статусы воркера («Переподключение…»)
            # не должны перекрыть уже показанный статус «Выключено».
            return
        runtime.last_status = status
        self.stream_status_changed.emit(stream_id, status)
        lower = status.lower()
        if any(key in lower for key in ("ошибка", "нет сигнала", "переподключение")):
            logger.warning("Поток %s: %s", stream_id, status)
        else:
            logger.debug("Поток %s: %s", stream_id, status)

    def _on_fps(self, stream_id: str, fps: float) -> None:
        runtime = self._runtimes.get(stream_id)
        if runtime is None:
            return
        runtime.fps = fps
        self.stream_fps_changed.emit(stream_id, fps)

    def _on_finished(self, stream_id: str) -> None:
        # Воркер реально завершился: снимаем ссылки (уничтожение
        # завершённого QThread безопасно) и чистим список «останавливаемых».
        runtime = self._runtimes.get(stream_id)
        if (
            runtime is not None
            and runtime.worker is not None
            and not runtime.worker.isRunning()
        ):
            runtime.worker = None
            runtime.stopping = False
        # Завершившиеся воркеры в _stopping_workers больше не нужны —
        # держали их только живыми до finished.
        self._stopping_workers = {
            w for w in self._stopping_workers if w.isRunning()
        }
        self.stream_finished.emit(stream_id)

    def _on_result(self, stream_id: str, result: FrameResult) -> None:
        """Агрегирует детекции кадра в ClientTrack и обновляет состояние."""
        runtime = self._runtimes.get(stream_id)
        if runtime is None:
            return

        now = time.time()
        for det in result.detections:
            if det.track_id is None:
                continue
            tid = det.track_id
            track = runtime.tracks.get(tid)
            if track is None:
                track = ClientTrack(track_id=tid, first_seen=now, last_seen=now)
                runtime.tracks[tid] = track
            track.update(det.emotion, det.emotion_confidence)

        # Чистим треки, которые давно не появлялись — считаем, что клиент ушёл.
        stale = [
            tid for tid, tr in runtime.tracks.items()
            if now - tr.last_seen > TRACK_TTL_SECONDS
        ]
        for tid in stale:
            del runtime.tracks[tid]

        self.tracks_updated.emit(stream_id, list(runtime.tracks.values()))

    # --- Производные метрики для UI -----------------------------------------

    @staticmethod
    def summarize_tracks(tracks: List[ClientTrack]) -> Dict[str, int]:
        """Считает количества клиентов по статусам."""
        summary = {STATUS_INCIDENT: 0, STATUS_UNSURE: 0, STATUS_NORMAL: 0}
        for tr in tracks:
            status = tr.status
            if status in summary:
                summary[status] += 1
            else:
                summary[STATUS_NORMAL] += 1
        return summary

    def is_live(self) -> bool:
        """True, если хотя бы один поток сейчас в эфире."""
        return any(rt.is_live for rt in self._runtimes.values())
