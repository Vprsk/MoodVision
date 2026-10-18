"""Единая визуальная тема приложения «EmoSense» (тёмная, по референсам).

Содержит палитру цветов, стили Qt (QSS) и цвета статусов/эмоций для UI.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

# --- Базовая палитра -------------------------------------------------------

BG_DARKEST = "#0b1020"      # самый тёмный фон (корень)
BG_BASE = "#0e1526"         # основной фон панелей
BG_CARD = "#141c30"         # фон карточек/плиток
BG_CARD_HOVER = "#1b2540"   # подсветка карточки
BG_ELEVATED = "#182034"    # фон всплывающих панелей
BG_HEADER = "#0f172b"       # фон верхней панели
BORDER = "#24314f"          # цвет разделителей/рамок
BORDER_STRONG = "#33436b"

TEXT_PRIMARY = "#e8edf7"
TEXT_SECONDARY = "#93a0bb"
TEXT_MUTED = "#5f6b85"

ACCENT = "#3b82f6"          # основной акцент (синий)
ACCENT_HOVER = "#2f6fe0"
ACCENT_SOFT = "#1d3a6b"

# --- Статусы ---------------------------------------------------------------

STATUS_LIVE = "#22c55e"        # «В эфире» — зелёный
STATUS_WARN = "#f59e0b"        # «Подключение/Не уверена» — жёлтый
STATUS_INCIDENT = "#f97316"    # «Инцидент» — оранжевый
STATUS_DANGER = "#ef4444"      # «Нет сигнала/ошибка» — красный
STATUS_IDLE = "#64748b"        # «Выключено/нет данных» — серый

# --- Цвета эмоций (для UI, в HEX, под тон рамок из annotate.py) ------------

EMOTION_HEX: dict[str, str] = {
    "neutral": "#c8d0dd",
    "happy": "#eab308",
    "sad": "#3b82f6",
    "angry": "#ef4444",
    "surprise": "#d946ef",
    "fear": "#a855f7",
    "disgust": "#22c55e",
    "unknown": "#f59e0b",
}


def emotion_color(name: str) -> str:
    """Возвращает HEX-цвет для эмоции (fallback — серый)."""
    return EMOTION_HEX.get(name, TEXT_MUTED)


def status_color(status: str) -> str:
    """Возвращает HEX-цвет для текстового статуса потока."""
    mapping = {
        "В эфире": STATUS_LIVE,
        "Подключение…": STATUS_WARN,
        "Не уверена": STATUS_WARN,
        "Инцидент": STATUS_INCIDENT,
        "Нет сигнала": STATUS_DANGER,
        "Выключено": STATUS_IDLE,
    }
    if status in mapping:
        return mapping[status]
    if status.startswith("Ошибка") or status.startswith("Поток завершён"):
        return STATUS_DANGER
    return TEXT_SECONDARY


def mix(hex_a: str, hex_b: str, ratio: float) -> str:
    """Смешивает два HEX-цвета. ratio=0 -> a, ratio=1 -> b.

    Удобно для получения мягких подложек под яркими акцентами.
    """
    ca, cb = QColor(hex_a), QColor(hex_b)
    r = round(ca.red() + (cb.red() - ca.red()) * ratio)
    g = round(ca.green() + (cb.green() - ca.green()) * ratio)
    b = round(ca.blue() + (cb.blue() - ca.blue()) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def soft(hex_color: str, ratio: float = 0.12) -> str:
    """Мягкая подложка: смешивает цвет с базовым фоном (BG_CARD)."""
    return mix(hex_color, BG_CARD, ratio)


# --- Стили Qt (QSS) --------------------------------------------------------
# Определены в отдельном модуле ui/style_qss.py, чтобы не раздувать палитру.
