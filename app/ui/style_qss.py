"""Стили Qt (QSS) для тёмной темы «EmoSense».

Вынесены из theme.py, чтобы палитра оставалась чистой. Используют
константы цветов из app.theme.
"""

from app import theme as T

QSS = f"""
* {{
    font-family: 'Segoe UI', 'Inter', sans-serif;
    color: {T.TEXT_PRIMARY};
}}

QMainWindow, QWidget#RootWidget {{
    background: {T.BG_DARKEST};
}}

QWidget {{
    background: {T.BG_BASE};
}}

/* --- Верхняя панель --- */
QWidget#TopBar {{
    background: {T.BG_HEADER};
    border-bottom: 1px solid {T.BORDER};
}}
QLabel#AppTitle {{
    font-size: 17px;
    font-weight: 700;
    color: {T.TEXT_PRIMARY};
}}
QPushButton#NavButton {{
    background: transparent;
    color: {T.TEXT_SECONDARY};
    border: none;
    padding: 8px 14px;
    font-size: 14px;
    border-radius: 8px;
    text-align: left;
}}
QPushButton#NavButton:hover {{
    color: {T.TEXT_PRIMARY};
    background: {T.BG_CARD};
}}
QPushButton#NavButton:checked {{
    color: {T.TEXT_PRIMARY};
    background: {T.BG_CARD_HOVER};
}}

/* --- Основные кнопки --- */
QPushButton {{
    background: {T.BG_CARD};
    color: {T.TEXT_PRIMARY};
    border: 1px solid {T.BORDER};
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
}}
QPushButton:hover {{
    background: {T.BG_CARD_HOVER};
    border-color: {T.BORDER_STRONG};
}}
QPushButton:disabled {{
    color: {T.TEXT_MUTED};
    background: {T.BG_BASE};
}}
QPushButton#PrimaryButton {{
    background: {T.ACCENT};
    color: white;
    border: none;
    font-weight: 600;
}}
QPushButton#PrimaryButton:hover {{
    background: {T.ACCENT_HOVER};
}}
QPushButton#DangerButton {{
    background: {T.STATUS_DANGER};
    color: white;
    border: none;
}}

/* --- Поля ввода --- */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {T.BG_BASE};
    color: {T.TEXT_PRIMARY};
    border: 1px solid {T.BORDER};
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 13px;
    selection-background-color: {T.ACCENT};
}}
QLineEdit:focus, QComboBox:focus {{
    border-color: {T.ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {T.BG_CARD};
    color: {T.TEXT_PRIMARY};
    border: 1px solid {T.BORDER};
    selection-background-color: {T.ACCENT_SOFT};
}}

/* --- Карточки/панели --- */
QFrame#Card {{
    background: {T.BG_CARD};
    border: 1px solid {T.BORDER};
    border-radius: 12px;
}}
QFrame#CardSelected {{
    background: {T.BG_CARD};
    border: 2px solid {T.ACCENT};
    border-radius: 12px;
}}

/* --- Заголовки --- */
QLabel#SectionTitle {{
    font-size: 16px;
    font-weight: 700;
    color: {T.TEXT_PRIMARY};
}}
QLabel#PageTitle {{
    font-size: 22px;
    font-weight: 700;
    color: {T.TEXT_PRIMARY};
}}
QLabel#Muted {{
    color: {T.TEXT_SECONDARY};
    font-size: 12px;
}}

/* --- Скроллбары --- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {T.BORDER_STRONG};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {T.ACCENT};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {T.BORDER_STRONG};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0; height: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* --- Таблицы --- */
QTableView, QTableWidget {{
    background: {T.BG_CARD};
    alternate-background-color: {T.BG_BASE};
    gridline-color: {T.BORDER};
    border: 1px solid {T.BORDER};
    border-radius: 10px;
    selection-background-color: {T.ACCENT_SOFT};
}}
QHeaderView::section {{
    background: {T.BG_HEADER};
    color: {T.TEXT_SECONDARY};
    padding: 8px;
    border: none;
    border-bottom: 1px solid {T.BORDER};
    font-weight: 600;
}}
"""
