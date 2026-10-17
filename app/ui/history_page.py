"""Страница «История»: сохранённые результаты анализа видеофайлов.

Список анализов строится из хранилища (app.core.file_analysis.list_analyses):
каждый сохранённый анализ — раскрывающаяся карточка (QFrame#Card) в
вертикальном списке внутри QScrollArea. При разворачивании карточка
загружает полный лог (load_analysis) и показывает таблицу детекций
(кадр, bbox, эмоция, уверенность). Открыта одновременно только одна
карточка (аккордеон) — при разворачивании новой предыдущая сворачивается.

Метод refresh() перечитывает хранилище и перестраивает список, сохраняя
развёрнутое состояние по analysis_id; MainWindow вызывает его
автоматически после завершения анализа файла.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.core.file_analysis import list_analyses, load_analysis
from app.ui.widgets.badges import Badge


# ---------------------------------------------------------------------------
# Форматирование значений лога
# ---------------------------------------------------------------------------


def _format_date(iso: str) -> str:
    """Приводит ISO-дату к виду «ДД.ММ.ГГГГ ЧЧ:ММ» (fallback — срез строки)."""
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return iso[:19].replace("T", " ")


def _fmt_bbox(bbox) -> str:
    """Компактный вид bbox: [x1, y1, x2, y2] (целые — без дробной части)."""
    if not bbox:
        return "—"
    parts = []
    for value in bbox:
        try:
            number = float(value)
        except (TypeError, ValueError):
            parts.append(str(value))
            continue
        parts.append(str(int(number)) if number.is_integer() else f"{number:.1f}")
    return f"[{', '.join(parts)}]"


def _fmt_confidence(value) -> str:
    """Уверенность с двумя знаками после запятой; None — «—»."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Виджеты карточки
# ---------------------------------------------------------------------------


class _CardHeader(QFrame):
    """Кликабельная шапка карточки (развернуть/свернуть по клику)."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _AnalysisCard(QFrame):
    """Раскрывающаяся карточка одного сохранённого анализа.

    Свёрнутое состояние — шапка: имя файла, бейдж с числом детекций,
    дата анализа, стрелка-индикатор. При разворачивании карточка
    лениво загружает полный лог (load_analysis) и строит таблицу
    детекций. Сигнал toggled(analysis_id) уходит на страницу, которая
    решает, какая карточка открыта (аккордеон).
    """

    toggled = Signal(str)

    def __init__(self, analysis: dict, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._analysis = analysis
        self._expanded = False
        self._content_built = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Шапка (клик — развернуть/свернуть) ---
        header = _CardHeader()
        header.setObjectName("CardHeader")
        header.setCursor(Qt.PointingHandCursor)
        header.setToolTip(analysis.get("source_file") or self.analysis_id)
        header.setStyleSheet(
            f"""
            QFrame#CardHeader {{ background: transparent; border: none;
                border-radius: 10px; }}
            QFrame#CardHeader:hover {{ background: {T.BG_CARD_HOVER}; }}
            """
        )
        header.clicked.connect(lambda: self.toggled.emit(self.analysis_id))

        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 12, 16, 12)
        hl.setSpacing(12)

        name = QLabel(os.path.basename(analysis.get("source_file") or self.analysis_id))
        name.setWordWrap(True)
        name.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 15px; font-weight: 600;"
            " background: transparent; border: none;"
        )
        hl.addWidget(name, 1)

        detections = int(analysis.get("summary", {}).get("detections", 0) or 0)
        self._badge = Badge(str(detections), T.ACCENT if detections else T.TEXT_MUTED)
        self._badge.setToolTip("Всего детекций")
        hl.addWidget(self._badge)

        date = QLabel(_format_date(analysis.get("created_at", "")))
        date.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: 12px; background: transparent;"
            " border: none;"
        )
        hl.addWidget(date)

        self._arrow = QLabel("▸")
        self._arrow.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 14px; background: transparent;"
            " border: none;"
        )
        hl.addWidget(self._arrow)

        layout.addWidget(header)

        # --- Контент (строится при первом разворачивании) ---
        self._content = QWidget()
        self._content.setVisible(False)
        layout.addWidget(self._content)

    @property
    def analysis_id(self) -> str:
        return self._analysis.get("analysis_id", "")

    @property
    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Разворачивает/сворачивает карточку (лог грузится лениво)."""
        if self._expanded == expanded:
            return
        self._expanded = expanded
        self._arrow.setText("▾" if expanded else "▸")
        if expanded:
            self._build_content()
        self._content.setVisible(expanded)

    def _build_content(self) -> None:
        """Строит таблицу детекций из полного лога (один раз)."""
        if self._content_built:
            return
        self._content_built = True

        cl = QVBoxLayout(self._content)
        cl.setContentsMargins(16, 4, 16, 16)
        cl.setSpacing(10)

        log = load_analysis(self.analysis_id) or {}
        results = log.get("results", [])
        # Строка таблицы — одна детекция; кадры отсортированы по frame_id.
        rows: List[tuple] = []
        for frame in sorted(results, key=lambda r: r.get("frame_id", 0)):
            frame_id = frame.get("frame_id", 0)
            for det in frame.get("detections", []):
                rows.append((frame_id, det))

        if not rows:
            empty = QLabel("Детекции не обнаружены")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet(
                f"color: {T.TEXT_MUTED}; font-size: 13px; background: transparent;"
                " border: none; padding: 12px;"
            )
            cl.addWidget(empty)
        else:
            table = QTableWidget(len(rows), 4)
            table.setHorizontalHeaderLabels(["Кадр", "bbox", "Эмоция", "Уверенность"])
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setSelectionMode(QAbstractItemView.NoSelection)
            table.setFocusPolicy(Qt.NoFocus)
            table.setAlternatingRowColors(True)
            # Таблица растёт по контенту — скроллит родительский QScrollArea.
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            table.verticalHeader().setVisible(False)
            table.verticalHeader().setDefaultSectionSize(28)
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Fixed)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            header.setSectionResizeMode(2, QHeaderView.Fixed)
            header.setSectionResizeMode(3, QHeaderView.Fixed)
            table.setColumnWidth(0, 70)
            table.setColumnWidth(2, 120)
            table.setColumnWidth(3, 110)

            for row, (frame_id, det) in enumerate(rows):
                bbox = det.get("bbox") or []
                emotion = det.get("emotion")
                confidence = det.get("emotion_confidence")

                frame_item = QTableWidgetItem(str(frame_id))
                frame_item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, 0, frame_item)

                table.setItem(row, 1, QTableWidgetItem(_fmt_bbox(bbox)))

                emotion_item = QTableWidgetItem(emotion or "—")
                emotion_item.setTextAlignment(Qt.AlignCenter)
                if emotion:
                    emotion_item.setForeground(QColor(T.emotion_color(emotion)))
                table.setItem(row, 2, emotion_item)

                conf_item = QTableWidgetItem(_fmt_confidence(confidence))
                conf_item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, 3, conf_item)

            header_h = header.sizeHint().height() or 30
            table.setFixedHeight(
                header_h
                + table.verticalHeader().defaultSectionSize() * table.rowCount()
                + 4
            )
            cl.addWidget(table)

        if self._analysis.get("annotated_path"):
            note = QLabel("Видео с разметкой доступно во вкладке «Файлы»")
            note.setStyleSheet(
                f"color: {T.TEXT_MUTED}; font-size: 11px; background: transparent;"
                " border: none;"
            )
            cl.addWidget(note)


# ---------------------------------------------------------------------------
# Страница
# ---------------------------------------------------------------------------


class HistoryPage(QWidget):
    """Страница истории: раскрывающиеся карточки сохранённых анализов."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._cards: List[_AnalysisCard] = []
        self._expanded_id: Optional[str] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        title = QLabel("История")
        title.setObjectName("PageTitle")
        root.addWidget(title)

        subtitle = QLabel("Сохранённые результаты анализа видеофайлов")
        subtitle.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: 13px; background: transparent;"
            " border: none;"
        )
        root.addWidget(subtitle)

        # --- Список карточек в QScrollArea ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        self._cards_layout = QVBoxLayout(container)
        self._cards_layout.setContentsMargins(0, 0, 8, 0)
        self._cards_layout.setSpacing(12)
        self._cards_layout.addStretch(1)
        self.scroll.setWidget(container)
        root.addWidget(self.scroll, 1)

        # --- Пустое состояние ---
        self._empty = QLabel(
            "Пока нет результатов анализа…\nЗагрузите видео во вкладке «Файлы»"
        )
        self._empty.setWordWrap(True)
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setStyleSheet(
            f"color: {T.TEXT_MUTED}; font-size: 15px; background: transparent;"
            " border: none;"
        )
        root.addWidget(self._empty, 1)

        # Первичная загрузка списка.
        self.refresh()

    # --- Данные -------------------------------------------------------------

    def refresh(self) -> None:
        """Перечитывает list_analyses() и перестраивает список карточек.

        Развёрнутое состояние сохраняется по analysis_id, если анализ
        ещё существует. Вызывается автоматически из MainWindow после
        завершения анализа файла.
        """
        expanded_id = self._expanded_id
        self._clear_cards()
        found = False
        for analysis in list_analyses():
            card = _AnalysisCard(analysis, parent=self)
            card.toggled.connect(self._on_card_toggled)
            self._cards.append(card)
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
            if analysis.get("analysis_id") == expanded_id:
                card.set_expanded(True)
                found = True
        if not found:
            self._expanded_id = None
        self._update_empty_state()

    def _clear_cards(self) -> None:
        """Удаляет все карточки из layout (stretch в конце сохраняется)."""
        for card in self._cards:
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

    def _update_empty_state(self) -> None:
        """Переключает список карточек и placeholder «нет результатов»."""
        has_cards = bool(self._cards)
        self.scroll.setVisible(has_cards)
        self._empty.setVisible(not has_cards)

    # --- Поведение карточек -------------------------------------------------

    def _on_card_toggled(self, analysis_id: str) -> None:
        """Аккордеон: открыта одна карточка, остальные свёрнуты."""
        if self._expanded_id == analysis_id:
            # Клик по уже открытой карточке — сворачиваем её.
            self._expanded_id = None
            for card in self._cards:
                if card.analysis_id == analysis_id:
                    card.set_expanded(False)
                    break
            return
        self._expanded_id = analysis_id
        for card in self._cards:
            card.set_expanded(card.analysis_id == analysis_id)