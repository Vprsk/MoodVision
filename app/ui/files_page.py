"""Страница «Файлы»: пакетный анализ локальных видеофайлов.

Левая колонка — очередь файлов на анализ (drag-and-drop из проводника или
выбор через QFileDialog). Правая колонка — результат: аннотированное видео
(annotated.mp4 из папки результата, там уже наложены bbox и эмоция) в
переиспользуемом VideoPlayer + сводка по анализу (файл, дата, кадры,
детекции) + список уже проанализированных записей.

Анализ запускается последовательно (по одному файлу за раз — просто и
надёжно) через FileAnalysisWorker из app.core.file_analysis: модель
создаётся только внутри рабочего потока при start(). После каждого
успешного анализа эмитится сигнал analysis_completed — «История»
подписывается на него в MainWindow.

Для просмотра результата используется тот же паттерн, что в StreamsPage:
VideoPlayer получает кадры через сигнал frame_ready от QThread-источника.
Здесь таким источником выступает лёгкий _FrameReader (чтение файла через
cv2.VideoCapture без инференса), повторяющий API VideoWorker
(frame_ready / media_info / position_changed / paused_changed /
toggle_pause / seek), поэтому подключается к плееру тем же способом.
"""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.core.file_analysis import FileAnalysisWorker, list_analyses
from app.core.logging import get_logger
from app.ui.widgets.badges import Badge, MetricCard
from app.ui.widgets.video_player import VideoPlayer

logger = get_logger("files_page")

# Фильтр видео для диалога выбора файлов (расширения «и т.п.» во «Все файлы»).
_VIDEO_FILTER = (
    "Видео (*.mp4 *.avi *.mkv *.mov *.webm *.mpeg *.mpg *.wmv);;Все файлы (*)"
)


class _FileDropZone(QFrame):
    """Компактная зона перетаскивания видеофайлов (аналог _DropZone).

    Принимает файлы из проводника (drag-and-drop) и предоставляет кнопку
    «Выбрать файлы…». Список путей отдаётся через колбэк files_dropped,
    который назначает страница.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setAcceptDrops(True)
        self.setMinimumHeight(140)
        self.files_dropped = None  # устанавливается страницей

        self._normal = (
            f"background: {T.BG_CARD}; border: 2px dashed {T.BORDER_STRONG};"
            " border-radius: 12px;"
        )
        self.setStyleSheet(self._normal)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignCenter)

        icon = QLabel("⬆")
        icon.setStyleSheet(
            f"font-size: 28px; color: {T.TEXT_MUTED}; background: transparent;"
            " border: none;"
        )
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        text = QLabel("Перетащите видео сюда")
        text.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; background: transparent; border: none;"
        )
        text.setAlignment(Qt.AlignCenter)
        layout.addWidget(text)

        self.choose_btn = QPushButton("Выбрать файлы…")
        self.choose_btn.setObjectName("PrimaryButton")
        self.choose_btn.setFixedWidth(170)
        self.choose_btn.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.choose_btn, 0, Qt.AlignCenter)

        fmt = QLabel("MP4, AVI, MKV, MOV, WEBM")
        fmt.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: 11px; background: transparent;"
            " border: none;"
        )
        fmt.setAlignment(Qt.AlignCenter)
        layout.addWidget(fmt)

    # --- Drag-and-drop ---

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                f"background: {T.BG_CARD_HOVER}; border: 2px dashed {T.ACCENT};"
                " border-radius: 12px;"
            )

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.setStyleSheet(self._normal)

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.dragLeaveEvent(event)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths and self.files_dropped:
            self.files_dropped(paths)


class _FrameReader(QThread):
    """Читает видеофайл и отдаёт кадры в плеер (без инференса и модели).

    Повторяет API VideoWorker, с которым работает VideoPlayer
    (frame_ready / media_info / position_changed / paused_changed /
    toggle_pause / seek_by / seek_to_ratio), поэтому подключается к плееру
    тем же способом, что StreamsPage подключает воркер потока. Кадры
    читаются в собственном потоке, чтобы не блокировать GUI.
    """

    frame_ready = Signal(np.ndarray)
    media_info = Signal(float, float, bool)
    position_changed = Signal(float, float)
    paused_changed = Signal(bool)

    def __init__(self, path: str, parent=None) -> None:
        super().__init__(parent)
        self._path = str(path)
        self._running = False
        # Отдельный флаг остановки: request_stop() может прийти ДО того, как
        # run() выставит _running = True (поток ещё не стартовал). Раньше такой
        # запрос терялся, wait() не дожидался и deleteLater() уничтожал
        # работающий QThread — Qt аварийно завершал процесс (см. _stop_playback).
        self._stop_requested = False
        self._paused = False
        self._duration: float = 0.0
        self._pending_seek_ms: Optional[int] = None
        self._cap: Optional[cv2.VideoCapture] = None

    # --- Управление (вызывается из GUI-потока) ---

    def request_stop(self) -> None:
        """Просит поток остановиться (проверяется в цикле run)."""
        self._stop_requested = True
        self._running = False
        # Снимаем паузу, чтобы цикл не «залип» и увидел остановку.
        self._paused = False

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self.paused_changed.emit(self._paused)

    def seek_by(self, seconds: float) -> None:
        """Перематывает относительно текущей позиции (применится в run)."""
        if self._cap is None:
            return
        current_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        self._pending_seek_ms = max(0, int(current_ms + seconds * 1000))

    def seek_to_ratio(self, ratio: float) -> None:
        """Перематывает на долю длительности (0..1)."""
        if self._cap is None or self._duration <= 0.0:
            return
        self._pending_seek_ms = int(max(0.0, min(1.0, ratio)) * self._duration * 1000)

    # --- QThread ---

    def run(self) -> None:  # noqa: D102 (Qt-переопределение)
        self._running = True
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            cap.release()
            self._running = False
            logger.error("Не удалось открыть видео для просмотра: %s", self._path)
            return
        self._cap = cap

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 25.0
        total_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        self._duration = total_frames / fps if fps > 0.0 and total_frames > 0 else 0.0
        self.media_info.emit(fps, self._duration, True)
        frame_wait_ms = max(1, int(1000 / fps))

        try:
            # _stop_requested проверяется дополнительно: запрос остановки мог
            # прийти до старта потока (тогда run() только что выставил
            # _running = True и «перезатёр» бы запрос).
            while self._running and not self._stop_requested:
                # --- Пауза: кадры не читаем, но реагируем на перемотку. ---
                if self._paused:
                    self.msleep(30)
                    continue

                if self._pending_seek_ms is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, self._pending_seek_ms)
                    self._pending_seek_ms = None

                ret, frame = cap.read()
                if not ret:
                    break  # дошли до конца файла
                self.frame_ready.emit(frame.copy())
                current = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                self.position_changed.emit(current, self._duration)
                self.msleep(frame_wait_ms)
        finally:
            self._cap = None
            cap.release()
            self._running = False

        # Достигли конца — возвращаем плеер в состояние «пауза» (иконка play).
        self.paused_changed.emit(False)


class FilesPage(QWidget):
    """Страница пакетного анализа локальных видеофайлов."""

    # Путь к папке результата после каждого успешного анализа.
    analysis_completed = Signal(str)
    # Запрос перехода на вкладку «История» (кнопка «Открыть в Истории»).
    open_history_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._queue: List[str] = []
        # Ссылки на воркеров, чтобы их не собрал сборщик мусора.
        self._workers: List[FileAnalysisWorker] = []
        self._current_worker: Optional[FileAnalysisWorker] = None
        self._analyzing = False
        self._cancelled = False
        self._reader: Optional[_FrameReader] = None
        self._current_analysis: Optional[dict] = None
        # Автозапуск: анализ начинается сразу, как только в очередь добавлены
        # файлы (и сейчас ничего не анализируется). Флаг выключен только
        # тестовым скриптом files_ui_check.py для проверки самой очереди.
        self._auto_start = True

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_left())
        root.addWidget(self._build_right(), 1)

        # Связываем сигналы плеера со страницей ОДИН раз — делегирование
        # текущему ридеру (см. _on_play_pause и т.п.), чтобы при смене
        # видео не накапливать связи и не держать ссылки на мёртвые QThread.
        self.player.play_pause_clicked.connect(self._on_play_pause)
        self.player.seek_forward_clicked.connect(self._on_seek_forward)
        self.player.seek_backward_clicked.connect(self._on_seek_backward)
        self.player.seek_to_ratio.connect(self._on_seek_ratio)

        self._update_controls()
        # Показываем уже готовые анализы (при повторном открытии вкладки).
        self._reload_analyses()

    # --- Построение интерфейса ----------------------------------------------

    def _build_left(self) -> QWidget:
        """Левая колонка: выбор файлов, очередь, запуск анализа."""
        panel = QWidget()
        panel.setFixedWidth(400)
        panel.setStyleSheet(f"background: {T.BG_DARKEST};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Файлы")
        title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 22px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        layout.addWidget(title)

        self.drop = _FileDropZone()
        self.drop.choose_btn.clicked.connect(self._on_choose_files)
        # Drag-and-drop из проводника — колбэк страницы (как в HistoryPage).
        self.drop.files_dropped = self._on_files_dropped
        layout.addWidget(self.drop)

        queue_title = QLabel("Очередь анализа")
        queue_title.setObjectName("SectionTitle")
        layout.addWidget(queue_title)

        self.queue_list = QListWidget()
        self.queue_list.setStyleSheet(self._list_qss())
        self.queue_list.setToolTip("Файлы, ожидающие анализа")
        layout.addWidget(self.queue_list, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._remove_btn = QPushButton("Удалить выбранный")
        self._remove_btn.setCursor(Qt.PointingHandCursor)
        self._remove_btn.clicked.connect(self._on_remove_selected)
        row.addWidget(self._remove_btn)
        self._clear_btn = QPushButton("Очистить")
        self._clear_btn.setCursor(Qt.PointingHandCursor)
        self._clear_btn.clicked.connect(self._on_clear_queue)
        row.addWidget(self._clear_btn)
        layout.addLayout(row)

        self._analyze_btn = QPushButton("Анализировать")
        self._analyze_btn.setObjectName("PrimaryButton")
        self._analyze_btn.setCursor(Qt.PointingHandCursor)
        self._analyze_btn.clicked.connect(self._start_analysis)
        layout.addWidget(self._analyze_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self._status_label = QLabel("Готов к работе")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 12px; background: transparent;"
            " border: none;"
        )
        layout.addWidget(self._status_label)

        self._cancel_btn = QPushButton("Отменить")
        self._cancel_btn.setObjectName("DangerButton")
        self._cancel_btn.setCursor(Qt.PointingHandCursor)
        self._cancel_btn.setToolTip("Остановить анализ после текущего файла")
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._cancel_btn)

        return panel

    def _build_right(self) -> QWidget:
        """Правая колонка: сводка, плеер результата, список анализов."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # --- Заголовок результата + бейдж + кнопка в «Историю» ---
        head = QHBoxLayout()
        head.setSpacing(10)
        self._result_title = QLabel("Выберите анализ")
        self._result_title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 20px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        head.addWidget(self._result_title)
        self._result_badge = Badge("—", T.TEXT_MUTED)
        head.addWidget(self._result_badge)
        head.addStretch(1)
        self._history_btn = QPushButton("Открыть в Истории")
        self._history_btn.setCursor(Qt.PointingHandCursor)
        self._history_btn.setToolTip("Перейти к списку всех анализов")
        self._history_btn.clicked.connect(self.open_history_requested.emit)
        head.addWidget(self._history_btn)
        layout.addLayout(head)

        # --- Сводка по выбранному анализу ---
        meta = QHBoxLayout()
        meta.setSpacing(14)
        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        self._result_file = QLabel("Файл: —")
        self._result_file.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 12px; background: transparent;"
            " border: none;"
        )
        self._result_date = QLabel("Дата: —")
        self._result_date.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: 12px; background: transparent;"
            " border: none;"
        )
        info_col.addWidget(self._result_file)
        info_col.addWidget(self._result_date)
        meta.addLayout(info_col)
        self.card_frames = MetricCard("Кадров", "—", "с детекциями")
        meta.addWidget(self.card_frames)
        self.card_detections = MetricCard("Детекций", "—", "всего")
        meta.addWidget(self.card_detections)
        meta.addStretch(1)
        layout.addLayout(meta)

        # --- Плеер результата (annotated.mp4) ---
        self.player = VideoPlayer()
        layout.addWidget(self.player, 1)

        # --- Список проанализированных ---
        results_title = QLabel("Проанализированные")
        results_title.setObjectName("SectionTitle")
        layout.addWidget(results_title)

        self.results_list = QListWidget()
        self.results_list.setStyleSheet(self._list_qss())
        self.results_list.setFixedHeight(150)
        self.results_list.currentItemChanged.connect(self._on_result_selected)
        layout.addWidget(self.results_list)

        return panel

    @staticmethod
    def _list_qss() -> str:
        """Общий стиль списков страницы (как в HistoryPage)."""
        return f"""
            QListWidget {{ background: transparent; border: none; }}
            QListWidget::item {{ padding: 6px; }}
            QListWidget::item:selected {{ background: {T.BG_CARD_HOVER};
                border-radius: 8px; }}
            QListWidget::item:hover {{ background: {T.BG_CARD}; }}
        """

    # --- Выбор файлов и очередь ---------------------------------------------

    def _on_choose_files(self) -> None:
        """Открывает диалог выбора видеофайлов (несколько)."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выберите видеофайлы", "", _VIDEO_FILTER
        )
        if paths:
            self._add_files(paths)

    def _on_files_dropped(self, paths: List[str]) -> None:
        self._add_files(paths)

    def _add_files(self, paths: List[str]) -> None:
        """Добавляет новые пути в очередь (без дублей) и запускает анализ.

        Если автозапуск включён (``_auto_start``) и сейчас ничего не
        анализируется, пакет запускается автоматически — нажимать
        «Анализировать» не нужно. Если анализ уже идёт, новый файл просто
        дописывается в очередь и будет обработан следующим.
        """
        added = 0
        for path in paths:
            if path in self._queue:
                logger.debug("Файл уже в очереди, пропуск: %s", path)
                continue
            self._queue.append(path)
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            self.queue_list.addItem(item)
            added += 1
        if added:
            logger.info(
                "Добавлено в очередь %d файл(ов); всего в очереди: %d",
                added,
                len(self._queue),
            )
            self._set_status(f"В очереди: {len(self._queue)} файл(а)")
            self._update_controls()
            # Автозапуск пакета, если автозапуск включён и анализ не идёт.
            if self._auto_start and not self._analyzing:
                self._start_analysis()

    def _on_remove_selected(self) -> None:
        """Удаляет выбранный файл из очереди."""
        row = self.queue_list.currentRow()
        if row < 0:
            return
        item = self.queue_list.takeItem(row)
        path = item.data(Qt.UserRole)
        if path in self._queue:
            self._queue.remove(path)
        self._set_status(f"В очереди: {len(self._queue)} файл(а)")
        self._update_controls()

    def _on_clear_queue(self) -> None:
        """Очищает всю очередь файлов."""
        self.queue_list.clear()
        self._queue.clear()
        self._set_status("Очередь очищена")
        self._update_controls()

    # --- Запуск анализа ------------------------------------------------------

    def _start_analysis(self) -> None:
        """Запускает последовательный анализ всех файлов из очереди."""
        if self._analyzing or not self._queue:
            return
        self._analyzing = True
        self._cancelled = False
        self.progress.setValue(0)
        logger.info("Запуск пакетного анализа: %d файл(ов) в очереди", len(self._queue))
        self._set_status("Анализ запущен…", T.STATUS_LIVE)
        self._update_controls()
        self._launch_next()

    def _launch_next(self) -> None:
        """Запускает воркер для следующего файла очереди."""
        if not self._queue:
            self._finish_batch()
            return
        path = self._queue[0]
        logger.info(
            "Анализ файла (%d осталось в очереди): %s", len(self._queue), path
        )
        self._set_status(f"Анализ: {os.path.basename(path)}")
        worker = FileAnalysisWorker(path, parent=self)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(
            lambda result_dir, w=worker: self._on_analysis_finished(w, result_dir)
        )
        worker.failed.connect(
            lambda error, w=worker: self._on_analysis_failed(w, error)
        )
        self._workers.append(worker)
        self._current_worker = worker
        worker.start()

    def _pop_current_file(self) -> None:
        """Убирает первый (только что обработанный) файл из очереди."""
        if self._queue:
            self._queue.pop(0)
        if self.queue_list.count():
            self.queue_list.takeItem(0)

    def _on_progress(self, value: int) -> None:
        self.progress.setValue(value)

    def _on_analysis_finished(self, worker: FileAnalysisWorker, result_dir: str) -> None:
        """Успешное завершение анализа файла: обновляем UI и идём дальше."""
        self._release_worker(worker)
        self._pop_current_file()
        self.progress.setValue(100)
        self._set_status(f"Готово: {os.path.basename(result_dir)}", T.STATUS_LIVE)
        logger.info("Анализ завершён: %s", result_dir)

        # Обновляем список анализов и сразу показываем результат в плеере.
        self._reload_analyses()
        self._open_result_dir(result_dir)
        # «История» подписывается на этот сигнал в MainWindow.
        self.analysis_completed.emit(result_dir)

        if self._cancelled:
            # Отмена — останавливаемся, оставшуюся очередь не трогаем.
            self._finish_batch()
            return
        self._launch_next()

    def _on_analysis_failed(self, worker: FileAnalysisWorker, error: str) -> None:
        """Ошибка анализа файла: показываем сообщение и идём к следующему."""
        self._release_worker(worker)
        self._pop_current_file()
        self._set_status(f"Ошибка: {error}", T.STATUS_DANGER)
        # Подробности (traceback) уже залогированы в FileAnalysisWorker.run.
        logger.error("Ошибка анализа файла: %s", error)
        if self._cancelled:
            self._finish_batch()
            return
        self._launch_next()

    def _release_worker(self, worker: FileAnalysisWorker) -> None:
        """Освобождает завершённый воркер (поток уже окончил run())."""
        if self._current_worker is worker:
            self._current_worker = None
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()

    def _finish_batch(self) -> None:
        """Сбрасывает состояние анализа (очередь обработана или отменена)."""
        logger.info(
            "Пакетный анализ завершён (%s)",
            "отменён пользователем" if self._cancelled else "очередь пуста",
        )
        self._analyzing = False
        self._cancelled = False
        self._current_worker = None
        self._update_controls()

    def _on_cancel(self) -> None:
        """Отменяет анализ: останавливает текущий воркер через request_stop()."""
        if not self._analyzing:
            return
        logger.info("Запрошена отмена пакетного анализа")
        self._cancelled = True
        self._set_status("Отмена… (дожидаемся завершения текущего файла)", T.STATUS_WARN)
        if self._current_worker is not None:
            self._current_worker.request_stop()
        self._update_controls()

    # --- Результаты и плеер --------------------------------------------------

    def _reload_analyses(self) -> None:
        """Перечитывает сохранённые анализы (list_analyses) в список справа."""
        current_row = self.results_list.currentRow()
        self.results_list.clear()
        for analysis in list_analyses():
            source = os.path.basename(analysis.get("source_file") or "")
            created = (analysis.get("created_at") or "").replace("T", " ")
            item = QListWidgetItem(f"{source}  ·  {created[:19]}")
            item.setData(Qt.UserRole, analysis)
            item.setToolTip(analysis.get("annotated_path") or "")
            self.results_list.addItem(item)
        if self.results_list.count():
            # Пытаемся сохранить прежний выбор, иначе выбираем первый.
            row = min(current_row, self.results_list.count() - 1)
            if row < 0:
                row = 0
            # Глушим currentItemChanged: setCurrentRow сам вызвал бы
            # _on_result_selected → _open_analysis, а прямой вызов ниже
            # запустил бы ВТОРОЙ _FrameReader — двойной перезапуск плеера
            # ронял процесс (гонка остановки свежезапущенного потока).
            self.results_list.blockSignals(True)
            self.results_list.setCurrentRow(row)
            self.results_list.blockSignals(False)
            analysis = self.results_list.item(row).data(Qt.UserRole)
            if analysis:
                self._open_analysis(analysis)
        else:
            self._reset_result_panel()

    def _reset_result_panel(self) -> None:
        """Сбрасывает панель результата, когда анализов нет."""
        self._current_analysis = None
        self._result_title.setText("Выберите анализ")
        self._result_badge.set_badge("—", T.TEXT_MUTED)
        self._result_file.setText("Файл: —")
        self._result_date.setText("Дата: —")
        self.card_frames.set_value("—")
        self.card_detections.set_value("—")
        self.player.set_placeholder("Результат анализа")
        self.player.set_playing(False)

    def _on_result_selected(self, current, _previous) -> None:
        """Показывает выбранный анализ в плеере и сводке."""
        if current is None:
            return
        analysis = current.data(Qt.UserRole)
        if analysis:
            self._open_analysis(analysis)

    def _open_result_dir(self, result_dir: str) -> None:
        """Открывает в плеере анализ по пути к папке результата."""
        analysis_id = os.path.basename(os.path.normpath(result_dir))
        for i in range(self.results_list.count()):
            analysis = self.results_list.item(i).data(Qt.UserRole)
            if analysis and analysis.get("analysis_id") == analysis_id:
                self.results_list.setCurrentRow(i)
                return
        # Запасной вариант: анализ появился, но не попал в список.
        for analysis in list_analyses():
            if analysis.get("analysis_id") == analysis_id:
                self._open_analysis(analysis)
                return

    def _open_analysis(self, analysis: dict) -> None:
        """Показывает сводку и запускает просмотр annotated.mp4."""
        self._current_analysis = analysis
        source = os.path.basename(analysis.get("source_file") or "")
        created = (analysis.get("created_at") or "").replace("T", " ")
        summary = analysis.get("summary", {})

        self._result_title.setText(source or analysis.get("analysis_id", "Анализ"))
        self._result_file.setText(f"Файл: {source or '—'}")
        self._result_date.setText(f"Дата: {created or '—'}")
        self.card_frames.set_value(
            str(summary.get("frames", 0)), "кадров с детекциями"
        )
        self.card_detections.set_value(
            str(summary.get("detections", 0)), "всего детекций"
        )

        annotated = analysis.get("annotated_path")
        if annotated:
            self._result_badge.set_badge("ГОТОВО", T.STATUS_LIVE)
            self._start_playback(annotated)
        else:
            self._result_badge.set_badge("НЕТ ВИДЕО", T.STATUS_WARN)
            self.player.set_placeholder("Аннотированное видео отсутствует")
            self.player.set_playing(False)

    def _start_playback(self, path: str) -> None:
        """Открывает видеофайл в плеере через _FrameReader."""
        self._stop_playback()
        logger.debug("Воспроизведение результата: %s", path)
        self.player.set_placeholder("Загрузка…")

        reader = _FrameReader(path, parent=self)
        reader.frame_ready.connect(self.player.update_frame)
        reader.position_changed.connect(self.player.set_position)
        reader.media_info.connect(self.player.set_media_info)
        reader.paused_changed.connect(self.player.set_playing)
        # После старта потока плеер переходит в состояние «играет».
        reader.media_info.connect(
            lambda _fps, _dur, _seek: self.player.set_playing(True)
        )
        self._reader = reader
        reader.start()

    def _stop_playback(self) -> None:
        """Останавливает текущий ридер (если он был) и освобождает ресурсы."""
        reader = self._reader
        self._reader = None
        if reader is None:
            return
        reader.request_stop()
        if not reader.wait(2000):
            # Поток не успел завершиться — НЕ уничтожаем работающий QThread
            # (Qt аварийно завершает процесс: «QThread: Destroyed while thread
            # is still running»). Откладываем удаление до фактического конца.
            logger.warning(
                "Ридер %s не остановился за 2 с — отложенное удаление", reader
            )
            reader.finished.connect(reader.deleteLater)
            return
        reader.deleteLater()

    # --- Делегирование плеера текущему ридеру --------------------------------

    def _on_play_pause(self) -> None:
        if self._reader is not None:
            self._reader.toggle_pause()

    def _on_seek_forward(self) -> None:
        if self._reader is not None:
            self._reader.seek_by(VideoPlayer.SEEK_STEP_SECONDS)

    def _on_seek_backward(self) -> None:
        if self._reader is not None:
            self._reader.seek_by(-VideoPlayer.SEEK_STEP_SECONDS)

    def _on_seek_ratio(self, ratio: float) -> None:
        if self._reader is not None:
            self._reader.seek_to_ratio(ratio)

    # --- Общее ---------------------------------------------------------------

    def _update_controls(self) -> None:
        """Включает/выключает кнопки по состоянию очереди и анализа."""
        has_queue = bool(self._queue)
        self._analyze_btn.setEnabled(has_queue and not self._analyzing)
        self._cancel_btn.setEnabled(self._analyzing)
        self._remove_btn.setEnabled(has_queue and not self._analyzing)
        self._clear_btn.setEnabled(has_queue and not self._analyzing)

    def _set_status(self, text: str, color: str = "") -> None:
        """Обновляет статусную строку (опционально — цвет текста)."""
        self._status_label.setText(text)
        if color:
            style = (
                f"color: {color}; font-size: 12px; background: transparent;"
                " border: none;"
            )
        else:
            style = (
                f"color: {T.TEXT_SECONDARY}; font-size: 12px; background:"
                " transparent; border: none;"
            )
        self._status_label.setStyleSheet(style)

    def shutdown(self) -> None:
        """Останавливает воркеры и плеер (вызывается при закрытии окна).

        Не даёт QThread-объектам быть уничтоженными во время выполнения.
        Модель при этом не участвует: FileAnalysisWorker останавливается
        через request_stop(), а _FrameReader — через request_stop().
        """
        worker = self._current_worker
        if worker is not None and worker.isRunning():
            logger.info("Остановка текущего анализа при закрытии страницы")
            worker.request_stop()
            if not worker.wait(10000):
                # Не успел остановиться — НЕ уничтожаем работающий QThread
                # (иначе Qt аварийно завершает процесс). Отсоединяем от
                # страницы и удаляем по факту завершения.
                logger.warning(
                    "Воркер анализа %s не остановился за 10 с — отложенное"
                    " удаление",
                    worker,
                )
                worker.setParent(None)
                worker.finished.connect(worker.deleteLater)
        self._stop_playback()