"""Плеер для показа видео с управлением воспроизведением.

Обёртка над VideoSurface, добавляющая панель управления:
  * play / pause;
  * перемотка назад / вперёд (для seekable-источников — видеофайлов);
  * прогресс-бар с текущим и общим временем.

Плеер НЕ управляет VideoWorker напрямую. Он лишь эмитит сигналы намерений
(play_pause_clicked, seek_clicked и т.п.), а StreamsPage связывает их с
методами активного воркера. Это сохраняет разделение ответственности и
позволяет переиспользовать плеер.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.ui.icons import (
    make_pause_icon,
    make_play_icon,
    make_seek_back_icon,
    make_seek_forward_icon,
)
from app.ui.widgets.video_surface import VideoSurface


def _fmt_time(seconds: float) -> str:
    """Форматирует секунды в MM:SS (или H:MM:SS для длинных видео)."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class VideoPlayer(QWidget):
    """Композит: видео + панель управления воспроизведением."""

    # Намерения пользователя (связываются с активным воркером в StreamsPage).
    play_pause_clicked = Signal()
    seek_forward_clicked = Signal()
    seek_backward_clicked = Signal()
    seek_to_ratio = Signal(float)  # 0..1 — позиция на прогресс-баре

    SEEK_STEP_SECONDS = 10.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._duration: float = 0.0
        self._seekable: bool = False
        self._playing: bool = False
        # Флаг: пользователь тащит ползунок — не перезаписываем его позицией.
        self._scrubbing: bool = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # --- Видео ---
        self.surface = VideoSurface("Кадр видеопотока")
        root.addWidget(self.surface, 1)

        # --- Панель управления ---
        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 4, 0)
        controls.setSpacing(8)

        self._play_btn = QPushButton()
        self._play_btn.setObjectName("PlayerButton")
        self._play_btn.setFixedSize(40, 40)
        self._play_btn.setCursor(Qt.PointingHandCursor)
        self._play_btn.setToolTip("Пуск / Пауза")
        self._play_btn.setIcon(make_play_icon(20))
        self._play_btn.clicked.connect(self.play_pause_clicked.emit)
        controls.addWidget(self._play_btn)

        self._back_btn = QPushButton()
        self._back_btn.setObjectName("PlayerButton")
        self._back_btn.setFixedSize(40, 40)
        self._back_btn.setCursor(Qt.PointingHandCursor)
        self._back_btn.setToolTip(f"Назад {int(self.SEEK_STEP_SECONDS)} с")
        self._back_btn.setIcon(make_seek_back_icon(20))
        self._back_btn.clicked.connect(self.seek_backward_clicked.emit)
        controls.addWidget(self._back_btn)

        self._fwd_btn = QPushButton()
        self._fwd_btn.setObjectName("PlayerButton")
        self._fwd_btn.setFixedSize(40, 40)
        self._fwd_btn.setCursor(Qt.PointingHandCursor)
        self._fwd_btn.setToolTip(f"Вперёд {int(self.SEEK_STEP_SECONDS)} с")
        self._fwd_btn.setIcon(make_seek_forward_icon(20))
        self._fwd_btn.clicked.connect(self.seek_forward_clicked.emit)
        controls.addWidget(self._fwd_btn)

        self._time_now = QLabel("00:00")
        self._time_now.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 12px;"
            " background: transparent; border: none;"
        )
        controls.addWidget(self._time_now)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setSingleStep(10)
        self._slider.setPageStep(50)
        self._slider.setCursor(Qt.PointingHandCursor)
        self._slider.setStyleSheet(self._slider_qss())
        self._slider.sliderPressed.connect(self._on_scrub_start)
        self._slider.sliderReleased.connect(self._on_scrub_end)
        controls.addWidget(self._slider, 1)

        self._time_total = QLabel("--:--")
        self._time_total.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; font-size: 12px;"
            " background: transparent; border: none;"
        )
        controls.addWidget(self._time_total)

        root.addLayout(controls)

    @staticmethod
    def _slider_qss() -> str:
        return f"""
            QSlider::groove:horizontal {{
                background: {T.BG_DARKEST};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: {T.ACCENT};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {T.ACCENT};
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
        """

    # --- Обновление из воркера ---------------------------------------------

    def update_frame(self, frame_bgr) -> None:
        """Показывает новый кадр (делегирует в VideoSurface)."""
        self.surface.update_frame(frame_bgr)

    def set_position(self, current: float, duration: float) -> None:
        """Обновляет прогресс-бар и таймкоды по позиции из воркера."""
        if duration > 0:
            self._time_total.setText(_fmt_time(duration))
        if not self._scrubbing:
            ratio = (current / duration) if duration > 0 else 0.0
            self._slider.setValue(int(max(0.0, min(1.0, ratio)) * 1000))
        self._time_now.setText(_fmt_time(current))

    def set_media_info(self, src_fps: float, duration: float, seekable: bool) -> None:
        """Настраивает плеер под параметры источника."""
        self._duration = duration
        self._seekable = seekable
        self._time_total.setText(_fmt_time(duration) if duration > 0 else "LIVE")
        # Для live-источников перемотка и прогресс-бар недоступны.
        self._slider.setEnabled(seekable and duration > 0)
        self._back_btn.setEnabled(seekable)
        self._fwd_btn.setEnabled(seekable)

    def set_playing(self, playing: bool) -> None:
        """Обновляет иконку play/pause под текущее состояние."""
        self._playing = playing
        self._play_btn.setIcon(make_pause_icon(20) if playing else make_play_icon(20))

    def set_placeholder(self, text: str) -> None:
        """Показывает текстовую заглушку вместо кадра."""
        self.surface.set_placeholder(text)

    # --- Обработчики ползунка ----------------------------------------------

    def _on_scrub_start(self) -> None:
        self._scrubbing = True

    def _on_scrub_end(self) -> None:
        self._scrubbing = False
        ratio = self._slider.value() / 1000.0
        self.seek_to_ratio.emit(ratio)

