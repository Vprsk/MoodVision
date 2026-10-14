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
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Signal

from app.core.stream_config import (
    STATUS_INCIDENT,
    STATUS_NORMAL,
    STATUS_UNSURE,
    ClientTrack,
    StreamConfig,
)
from app.core.video_worker import VideoWorker
from contracts.schemas import FrameResult

# Сколько секунд трек не появляется в кадре, чтобы считаться «ушёдшим».
TRACK_TTL_SECONDS = 10.0


@dataclass
class StreamRuntime:
    """Среднее время выполнения одного потока: конфиг + воркер + треки."""

    config: StreamConfig
    worker: Optional[VideoWorker] = None
    tracks: Dict[int, ClientTrack] = field(default_factory=dict)
    last_status: str = "Выключено"
    fps: float = 0.0

    @property
    def is_live(self) -> bool:
        return self.worker is not None and self.worker.isRunning()


class StreamManager(QObject):
    """Управляет всеми видеопотоками приложения."""

    # Сигналы для UI: обновление конкретного потока.
    stream_status_changed = Signal(str, str)      # stream_id, status
    stream_fps_changed = Signal(str, float)       # stream_id, fps
    tracks_updated = Signal(str, list)            # stream_id, list[ClientTrack]
    incident_raised = Signal(str, str)            # stream_id, описание
    stream_finished = Signal(str)                 # stream_id (поток завершился)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._runtimes: Dict[str, StreamRuntime] = {}

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

        worker = VideoWorker(runtime.config)
        worker.frame_ready.connect(self._on_frame_ready)
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
        return True

    def stop_stream(self, stream_id: str) -> None:
        """Останавливает воркер потока (мягко просит завершиться)."""
        runtime = self._runtimes.get(stream_id)
        if runtime is None or runtime.worker is None:
            return
        runtime.worker.stop()
        runtime.worker.wait(3000)
        runtime.worker = None
        runtime.last_status = "Выключено"
        runtime.fps = 0.0
        self.stream_status_changed.emit(stream_id, runtime.last_status)

    def remove_stream(self, stream_id: str) -> None:
        """Останавливает и удаляет поток из менеджера."""
        self.stop_stream(stream_id)
        self._runtimes.pop(stream_id, None)

    def stop_all(self) -> None:
        """Останавливает все потоки (при закрытии приложения)."""
        for stream_id in list(self._runtimes.keys()):
            self.stop_stream(stream_id)

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

    def _on_status(self, stream_id: str, status: str) -> None:
        runtime = self._runtimes.get(stream_id)
        if runtime is None:
            return
        runtime.last_status = status
        self.stream_status_changed.emit(stream_id, status)

    def _on_fps(self, stream_id: str, fps: float) -> None:
        runtime = self._runtimes.get(stream_id)
        if runtime is None:
            return
        runtime.fps = fps
        self.stream_fps_changed.emit(stream_id, fps)

    def _on_finished(self, stream_id: str) -> None:
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
        return self.worker is not None and self.worker.isRunning()
