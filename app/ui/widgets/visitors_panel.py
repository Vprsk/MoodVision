"""Панель «Посетители»: список активных треков с доминирующей эмоцией.

Каждая строка показывает: ID клиента, бейдж статуса, доминирующую эмоцию
со временем в кадре и полоску уверенности, окрашенную в цвет эмоции.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.core.stream_config import ClientTrack
from app.ui.widgets.badges import Badge


class _VisitorRow(QFrame):
    """Одна строка: клиент, эмоция, полоска уверенности."""

    def __init__(self, track: ClientTrack, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setStyleSheet(
            f"background: {T.BG_CARD}; border: 1px solid {T.BORDER};"
            " border-radius: 10px;"
        )
        self._track = track

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        # --- Верхняя строка: ID + бейдж статуса ---
        top = QHBoxLayout()
        top.setSpacing(10)
        id_label = QLabel(f"Клиент #{track.track_id}")
        id_label.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 14px; font-weight: 600;"
            " background: transparent; border: none;"
        )
        top.addWidget(id_label)
        top.addStretch(1)
        self._badge = Badge(track.status, T.status_color(track.status))
        top.addWidget(self._badge)
        root.addLayout(top)

        # --- Строка эмоции + уверенность ---
        emo = QHBoxLayout()
        emo.setSpacing(10)
        color = T.emotion_color(track.dominant_emotion.value)
        self._emo_label = QLabel(f"{self._emotion_ru(track.dominant_emotion.value)}"
                                 f" · {track.dominant_confidence:.2f}")
        self._emo_label.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 12px;"
            " background: transparent; border: none;"
        )
        emo.addWidget(self._emo_label, 1)
        self._conf_label = QLabel(f"{track.dominant_confidence:.2f}")
        self._conf_label.setStyleSheet(
            f"color: {color}; font-weight: 700; background: transparent;"
            " border: none;"
        )
        emo.addWidget(self._conf_label)
        root.addLayout(emo)

        # --- Полоска уверенности ---
        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setRange(0, 100)
        self._bar.setValue(int(track.dominant_confidence * 100))
        self._bar.setFixedHeight(6)
        self._bar.setStyleSheet(
            f"""
            QProgressBar {{ background: {T.BG_DARKEST}; border-radius: 3px; border: none; }}
            QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}
            """
        )
        root.addWidget(self._bar)

    @staticmethod
    def _emotion_ru(value: str) -> str:
        mapping = {
            "neutral": "Нейтрально",
            "happy": "Радость",
            "sad": "Грусть",
            "angry": "Злость",
            "surprise": "Удивление",
            "fear": "Страх",
            "disgust": "Отвращение",
            "unknown": "Не определено",
        }
        return mapping.get(value, value)


class VisitorsPanel(QWidget):
    """Скроллируемый список активных клиентов с сортировкой по статусу."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        header = QLabel("Посетители")
        header.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 16px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        root.addWidget(header)

        # Прокручиваемая область со списком строк.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")
        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._list_layout = QVBoxLayout(self._container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(8)
        self._list_layout.addStretch(1)
        scroll.setWidget(self._container)
        root.addWidget(scroll, 1)

    @staticmethod
    def _make_placeholder() -> QLabel:
        """Создаёт текстовую заглушку (новый объект на каждый вызов)."""
        ph = QLabel("Клиентов пока нет")
        ph.setStyleSheet(
            f"color: {T.TEXT_MUTED}; background: transparent; border: none;"
        )
        ph.setAlignment(Qt.AlignCenter)
        return ph

    def update_tracks(self, tracks: List[ClientTrack]) -> None:
        """Перерисовывает список клиентов (сортировка: инциденты сверху)."""
        self._clear()

        if not tracks:
            self._list_layout.insertWidget(0, self._make_placeholder())
            self._list_layout.addStretch(1)
            return

        status_rank = {"Инцидент": 0, "Не уверена": 1, "Норма": 2}
        ordered = sorted(
            tracks, key=lambda t: (status_rank.get(t.status, 3), -t.dominant_confidence)
        )
        for i, track in enumerate(ordered):
            self._list_layout.insertWidget(i, _VisitorRow(track))
        self._list_layout.addStretch(1)

    def _clear(self) -> None:
        # Удаляем строки корректно: takeAt уже отцепляет item от layout'а,
        # а deleteLater() снимет родителя и удалит виджет. Намеренно НЕ
        # вызываем setParent(None) — иначе виджет становится top-level
        # окном и при частых обновлениях (на каждый кадр) окна накапливаются.
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
