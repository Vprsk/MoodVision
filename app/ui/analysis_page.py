"""Страница «Анализ видео»: пакетный анализ загруженных записей.

Показывает пустое состояние с призывом добавить поток или
проанализировать запись. Является точкой входа для пакетной обработки
видеофайлов через выбранный пайплайн контракта.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import theme as T
from app.ui.icons import make_plus_icon


class AnalysisPage(QWidget):
    """Страница анализа видео (пакетная обработка записей)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(14)

        icon = QLabel("🎬")
        icon.setStyleSheet("font-size: 54px; background: transparent; border: none;")
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        title = QLabel("Потоков пока нет")
        title.setStyleSheet(
            f"color: {T.TEXT_PRIMARY}; font-size: 24px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        desc = QLabel(
            "Добавьте поток с IP-камеры, веб-камеры или по ссылке RTSP / HTTP — "
            "распознавание эмоций начнётся сразу после подключения."
        )
        desc.setWordWrap(True)
        desc.setMaximumWidth(560)
        desc.setStyleSheet(
            f"color: {T.TEXT_SECONDARY}; background: transparent; border: none;"
        )
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)

        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)
        buttons_layout.setAlignment(Qt.AlignCenter)

        add_btn = QPushButton("  Добавить поток")
        add_btn.setObjectName("PrimaryButton")
        add_btn.setFixedWidth(240)
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.setIcon(make_plus_icon(box=20, color="#ffffff"))
        add_btn.setIconSize(QSize(18, 18))
        add_btn.clicked.connect(self._on_add_stream)
        buttons_layout.addWidget(add_btn, 0, Qt.AlignCenter)

        analyze_btn = QPushButton("Проанализировать запись")
        analyze_btn.setFixedWidth(240)
        analyze_btn.clicked.connect(self._on_analyze)
        buttons_layout.addWidget(analyze_btn, 0, Qt.AlignCenter)

        layout.addLayout(buttons_layout)

    def _on_add_stream(self) -> None:
        """Открывает диалог добавления потока."""
        from app.ui.dialogs.add_stream_dialog import AddStreamDialog

        dialog = AddStreamDialog(self)
        dialog.exec()

    def _on_analyze(self) -> None:
        """Открывает диалог выбора видеофайла для пакетного анализа."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "", "Видео (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        QMessageBox.information(
            self,
            "Анализ видео",
            f"Выбран файл:\n{path}\n\n"
            "Пакетный анализ записей подключается к контроллеру обработки "
            "и будет запущен через выбранный пайплайн.",
        )
