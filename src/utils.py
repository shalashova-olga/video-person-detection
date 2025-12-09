from typing import Tuple
from pathlib import Path
import os
import cv2


def open_video(path: str) -> cv2.VideoCapture:
    """Открыть видеофайл и проверить доступность; дать более явные причины ошибок.

    Args:
        path: Путь к видеофайлу.
    Returns:
        cv2.VideoCapture: Открытый источник видео.
    Raises:
        FileNotFoundError: Файл по пути не найден.
        PermissionError: Нет прав на чтение файла.
        RuntimeError: Файл найден, но не открылся.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if not p.is_file():
        raise FileNotFoundError(f"Ожидался файл, но это не файл: {path}")
    if not os.access(p, os.R_OK):
        raise PermissionError(f"Нет прав на чтение: {path}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(
            f"Не удалось открыть видео: {path}."
        )
    return cap


def read_video_props(cap: cv2.VideoCapture) -> Tuple[float, int, int]:
    """Считать (fps, width, height) у видеоисточника с безопасными дефолтами.

    Возвращает частоту кадров (fps), ширину (w) и высоту (h) кадра, чтобы выходной ролик получился корректным по длительности, темпу и геометрии.

    Args:
        cap (cv2.VideoCapture): Открытый источник видео.

    Returns:
        Tuple[float, int, int]: Кортеж `(fps, w, h)`, где
            - `fps` (float): частота кадров (> 0.0; иначе подставлен 25.0),
            - `w` (int): ширина кадра в пикселях (> 0),
            - `h` (int): высота кадра в пикселях (> 0).
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, w, h


def create_writer(out_path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """Создать MP4-писатель (cv2.VideoWriter) с безопасным фолбэком кодека.

    Функция гарантирует, что каталог под выходной файл существует, пытается
    открыть MP4-видеописатель с кодеком `mp4v`, а если не получилось —
    повторяет попытку с `avc1`. Если оба варианта не открылись, выбрасывает
    исключение с подсказкой проверить кодеки/систему.

    Args:
        out_path (str): Путь к выходному видеофайлу.
        fps (float): Частота кадров выходного видео (> 0).
        size (Tuple[int, int]): Размер кадра `(width, height)` в пикселях.

    Returns:
        cv2.VideoWriter: Готовый к записи объект видеописателя.

    Raises:
        RuntimeError: Не удалось открыть видеописатель ни с `mp4v`, ни с `avc1`.
    """
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)

    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_path, fourcc, fps, size)

    if not writer.isOpened():
        raise RuntimeError(
            "Не удалось открыть VideoWriter. Проверьте установленные кодеки или попробуйте другой FOURCC "
            "Убедитесь, что fps > 0 и размер кадра совпадает с size."
        )
    return writer

