"""Автономный анализ локальных видеофайлов: инференс + аннотированный mp4 + JSON-лог.

Модуль — ядро для вкладок «Файлы» и «История»:

  * FileAnalysisWorker — фоновый QThread: открывает файл, прогоняет кадры через
    модель (create_pipeline из pipeline_registry, как VideoWorker), пишет
    аннотированный mp4 (кодек mp4v, fps/размер как у исходника) и JSON-лог
    с детекциями по кадрам. Кадры в UI не эмитятся — это пакетная обработка.
  * Функции хранилища — единая конвенция расположения результатов анализа
    (data/analysis/<stem>_<YYYYmmdd-HHMMSS>/): save_log/load_log,
    list_analyses, load_analysis.
  * analyze_file_sync — синхронная обёртка для тестов и скриптов.

Модель создаётся ОДИН раз за анализ и только внутри рабочего потока
(YOLO/ByteTrack содержат состояние треков и не потокобезопасны) — тот же
механизм, что в VideoWorker._build_model.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from PySide6.QtCore import QThread, Signal

from app.core.annotate import draw_detection
from app.core.logging import get_logger
from contracts.schemas import FrameResult

logger = get_logger("file_analysis")

# Корень проекта: <root>/app/core/file_analysis.py -> parents[2] = <root>.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Имя аннотированного видео внутри папки результата.
ANNOTATED_FILENAME = "annotated.mp4"
# Имя JSON-лога внутри папки результата.
LOG_FILENAME = "log.json"

# Кодек для аннотированного mp4 (широко поддерживается плеерами).
_VIDEO_CODEC = "mp4v"
# FPS по умолчанию, если у источника не удалось прочитать.
_DEFAULT_FPS = 25.0

# Символы, недопустимые в имени папки на Windows.
_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*]')


# ---------------------------------------------------------------------------
# Хранилище результатов анализа
# ---------------------------------------------------------------------------


def analysis_storage_dir() -> Path:
    """Корневая директория результатов анализа (создаётся при необходимости).

    Единая конвенция: <корень проекта>/data/analysis/.
    """
    root = _PROJECT_ROOT / "data" / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sanitize_stem(stem: str) -> str:
    """Очищает имя файла от символов, недопустимых в имени папки (Windows)."""
    cleaned = _INVALID_FS_CHARS.sub("_", stem).strip(" .")
    return cleaned or "video"


def analysis_dir_for(file_path: str | Path, when: Optional[datetime] = None) -> Path:
    """Уникальная папка результата для конкретного файла.

    Формат: data/analysis/<stem>_<YYYYmmdd-HHMMSS>/. Параметр ``when``
    позволяет задать время явно (для детерминированных тестов).
    """
    stem = _sanitize_stem(Path(file_path).stem)
    ts = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return analysis_storage_dir() / f"{stem}_{ts}"


def save_log(analysis_dir: str | Path, payload: dict) -> Path:
    """Записывает JSON-лог анализа в папку результата. Возвращает путь к файлу.

    JSON пишется с ensure_ascii=False (кириллица читаемо) и indent=2.
    """
    folder = Path(analysis_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOG_FILENAME
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def load_log(analysis_dir: str | Path) -> dict:
    """Читает JSON-лог анализа. Возвращает {} если файла нет или он битый."""
    path = Path(analysis_dir) / LOG_FILENAME
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Не удалось прочитать лог %s: %s", path, exc)
        return {}


def list_analyses() -> list[dict]:
    """Список всех сохранённых анализов для «Истории».

    Сортировка по created_at (убывание). Каждый элемент:
      analysis_id, source_file, created_at, video, log_path,
      annotated_path (или None), summary {frames, detections}.
    """
    root = analysis_storage_dir()
    items: list[dict] = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        log_path = folder / LOG_FILENAME
        if not log_path.is_file():
            continue
        log = load_log(folder)
        if not log:
            continue
        annotated = folder / ANNOTATED_FILENAME
        results = log.get("results", [])
        total_detections = sum(len(r.get("detections", [])) for r in results)
        items.append(
            {
                "analysis_id": folder.name,
                "source_file": log.get("source_file", ""),
                "created_at": log.get("created_at", ""),
                "video": log.get("video", {}),
                "log_path": str(log_path),
                "annotated_path": str(annotated) if annotated.is_file() else None,
                "summary": {
                    "frames": len(results),
                    "detections": total_detections,
                },
            }
        )
    items.sort(key=lambda it: it["created_at"], reverse=True)
    return items


def load_analysis(analysis_id: str) -> Optional[dict]:
    """Загружает полный лог анализа по id (имени папки). None — если нет."""
    folder = analysis_storage_dir() / analysis_id
    if not folder.is_dir():
        return None
    log = load_log(folder)
    return log or None


# ---------------------------------------------------------------------------
# Приведение результатов контракта к плоскому JSON-логу
# ---------------------------------------------------------------------------


def _detection_to_dict(det) -> dict:
    """Приводит FaceDetection к плоскому dict для JSON-лога.

    bbox сохраняется в том же формате, что возвращает модель/схема:
    [x_min, y_min, x_max, y_max] — координаты углов исходного кадра.
    """
    return {
        "track_id": det.track_id,
        "bbox": [det.bbox.x_min, det.bbox.y_min, det.bbox.x_max, det.bbox.y_max],
        "emotion": det.emotion.value if det.emotion is not None else None,
        "face_confidence": det.face_confidence,
        "emotion_confidence": det.emotion_confidence,
    }


def _frame_result_to_dict(result: FrameResult) -> dict:
    """Приводит FrameResult к плоскому dict для JSON-лога."""
    return {
        "frame_id": result.frame_id,
        "detections": [_detection_to_dict(d) for d in result.detections],
    }


# ---------------------------------------------------------------------------
# Фоновый воркер анализа файла
# ---------------------------------------------------------------------------


class FileAnalysisWorker(QThread):
    """Фоновый анализ локального видеофайла (без показа кадров в UI).

    Сигналы:
      progress(int)    — процент обработанных кадров (0..100);
      finished_ok(str) — путь к папке результата, когда всё записано
                         (в т.ч. при отмене — сохраняется частичный лог);
      failed(str)      — текст ошибки.

    Модель создаётся один раз внутри потока через
    pipeline_registry.create_pipeline (как VideoWorker). Отмена — через
    request_stop(): цикл прерывается на ближайшей итерации, VideoWriter
    финализируется, частичный лог сохраняется.
    """

    progress = Signal(int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        file_path: str | Path,
        output_dir: Optional[str | Path] = None,
        pipeline_key: str = "yolo_vit",
        pipeline_params: Optional[dict] = None,
        parent: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self._file_path = str(file_path)
        self._output_dir = Path(output_dir) if output_dir else None
        self._pipeline_key = pipeline_key
        self._pipeline_params = dict(pipeline_params or {})
        self._running = False
        # Отдельный флаг остановки: request_stop() может прийти ДО того, как
        # run() выставит _running = True — иначе запрос терялся, и отмена
        # анализа не срабатывала.
        self._stop_requested = False
        self._result_dir: Optional[Path] = None
        self._error: Optional[str] = None

    # --- Управление (вызывается из GUI-потока) ---

    def request_stop(self) -> None:
        """Просит поток остановиться (проверяется в цикле на каждой итерации)."""
        self._stop_requested = True
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def result_dir(self) -> Optional[Path]:
        return self._result_dir

    @property
    def error(self) -> Optional[str]:
        return self._error

    # --- QThread ---

    def run(self) -> None:  # noqa: D102 (Qt-переопределение)
        self._running = True
        started = datetime.now()
        logger.info("Запуск анализа файла: %s", self._file_path)
        try:
            result_dir = self._analyze()
            self._result_dir = result_dir
            elapsed = (datetime.now() - started).total_seconds()
            logger.info(
                "Анализ файла завершён за %.1f с: %s", elapsed, result_dir
            )
            self.finished_ok.emit(str(result_dir))
        except Exception as exc:  # noqa: BLE001 — показываем любую ошибку анализа
            self._error = str(exc)
            logger.error(
                "Ошибка анализа файла %s: %s", self._file_path, exc, exc_info=True
            )
            self.failed.emit(str(exc))
        finally:
            self._running = False

    # --- Ядро анализа (вызывается из run или напрямую из analyze_file_sync) ---

    def _analyze(self) -> Path:
        """Выполняет анализ и возвращает путь к папке результата.

        При ошибке бросает исключение (run ловит и эмитит failed).
        """
        file_path = Path(self._file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        # --- Директория результата ---
        result_dir = self._output_dir or analysis_dir_for(file_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = result_dir / ANNOTATED_FILENAME

        # --- Открываем источник ---
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Не удалось открыть видеофайл: {file_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0.0:
            fps = _DEFAULT_FPS
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError(f"Не удалось определить размер кадра: {file_path}")
        duration_sec = total_frames / fps if total_frames > 0 else 0.0
        logger.info(
            "Источник открыт: %s (%dx%d, %.2f fps, кадров ~%d, %.1f с)",
            file_path.name,
            width,
            height,
            fps,
            total_frames,
            duration_sec,
        )

        # --- Создаём модель ОДИН раз (внутри потока, как VideoWorker) ---
        from app.core.pipeline_registry import create_pipeline

        logger.debug(
            "Создание модели пайплайна %r (параметры: %s)",
            self._pipeline_key,
            self._pipeline_params,
        )
        model_started = datetime.now()
        model = create_pipeline(self._pipeline_key, **self._pipeline_params)
        logger.info(
            "Модель %r создана за %.1f с", self._pipeline_key,
            (datetime.now() - model_started).total_seconds(),
        )

        # --- VideoWriter: кодек mp4v, fps и размер как у исходника ---
        fourcc = cv2.VideoWriter_fourcc(*_VIDEO_CODEC)
        writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Не удалось создать VideoWriter: {annotated_path}")

        results: list[dict] = []
        frame_id = 0
        # Логируем прогресс каждые 10% (на DEBUG), чтобы не засорять вывод.
        next_milestone = 10
        try:
            # _stop_requested проверяется дополнительно: запрос остановки мог
            # прийти до старта потока (тогда run() только что выставил
            # _running = True и «перезатёр» бы его).
            while self._running and not self._stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1

                result: FrameResult = model.predict(frame_id, frame)
                if result.detections:
                    results.append(_frame_result_to_dict(result))

                # Аннотируем копию кадра и пишем ВСЕ кадры — аннотированное
                # видео повторяет исходное по длительности и частоте кадров.
                annotated = frame.copy()
                for det in result.detections:
                    draw_detection(annotated, det)
                writer.write(annotated)

                if total_frames > 0:
                    percent = min(100, int(frame_id / total_frames * 100))
                else:
                    percent = 0
                if percent >= next_milestone:
                    logger.debug(
                        "Прогресс %s: %d%% (%d/%d кадров)",
                        file_path.name,
                        percent,
                        frame_id,
                        total_frames,
                    )
                    next_milestone += 10
                self.progress.emit(percent)
        finally:
            writer.release()
            cap.release()

        if self._stop_requested:
            logger.warning(
                "Анализ %s прерван пользователем на кадре %d из ~%d —"
                " сохраняется частичный результат",
                file_path.name,
                frame_id,
                total_frames,
            )

        # --- Лог анализа ---
        payload = {
            "source_file": str(file_path.resolve()),
            "created_at": datetime.now().astimezone().isoformat(),
            "video": {
                "fps": fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "duration_sec": round(duration_sec, 2),
            },
            "results": results,
        }
        save_log(result_dir, payload)
        logger.info(
            "Анализ %s: кадров с детекциями %d, лог %s",
            file_path.name,
            len(results),
            result_dir / LOG_FILENAME,
        )
        return result_dir


# ---------------------------------------------------------------------------
# Синхронная обёртка (для тестов и скриптов)
# ---------------------------------------------------------------------------


def analyze_file_sync(
    file_path: str | Path,
    output_dir: Optional[str | Path] = None,
    pipeline_key: str = "yolo_vit",
    pipeline_params: Optional[dict] = None,
) -> str:
    """Синхронный запуск анализа файла (для тестов и скриптов).

    Возвращает путь к папке результата или бросает исключение
    (исходный тип сохраняется). Выполняется в текущем потоке
    (без event loop) — удобно для проверок.
    """
    worker = FileAnalysisWorker(file_path, output_dir, pipeline_key, pipeline_params)
    worker._running = True
    try:
        return str(worker._analyze())
    finally:
        worker._running = False
