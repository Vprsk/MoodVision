from collections.abc import Iterator

import cv2 as cv
import numpy as np
from numpy.typing import NDArray


class CameraReader:
    """Захват видеопотока и последовательная выдача подготовленных кадров"""

    def __init__(self, source: str | int) -> None:
        self.source: str | int = source
        self._cap: cv.VideoCapture | None = None

    def _open_cap(self) -> None:
        """Открывает VideoCapture, если он ещё не открыт"""
        if self._cap is not None and self._cap.isOpened():
            return

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._cap = cv.VideoCapture(self.source)

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Не удалось открыть источник: {self.source}")

    def _cap_is_open(self) -> bool:
        """Проверяет, открыт ли видеопоток"""
        return self._cap is not None and self._cap.isOpened()

    def _close_cap(self) -> None:
        """Закрывает поток и освобождает ресурсы"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def preprocess(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Подготавливает кадр при необходимости"""
        return frame

    def get_frame(self) -> NDArray[np.uint8] | None:
        """Возвращает один кадр или None, если поток закончился"""
        self._open_cap()

        ok, frame = self._cap.read()
        if not ok:
            return None

        return self.preprocess(frame)

    def get_frames(self) -> Iterator[tuple[int, NDArray[np.uint8]]]:
        """Генератор: возвращает пары (номер кадра, кадр)"""
        frame_id = 0

        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    break

                yield frame_id, frame
                frame_id += 1

        finally:
            self._close_cap()
