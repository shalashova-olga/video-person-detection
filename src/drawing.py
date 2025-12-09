from typing import Tuple, Optional
import numpy as np
import cv2

COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9),
    (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4)
]

POSE_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0),
    (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170),
    (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255)
]


def score_to_color(score: Optional[float]) -> Tuple[int, int, int]:
    """Преобразовать уверенность [0..1] в цвет BGR по градиенту красный→зелёный.

    0.0  → красный  (0, 0, 255)
    0.5  → жёлто-оранжевый
    1.0  → зелёный  (36, 255, 12)

    Args:
        score: Уверенность модели. Если None — возвращается зелёный по умолчанию.

    Returns:
        Tuple[int, int, int]: Цвет в формате BGR.
    """
    if score is None:
        return (36, 255, 12)

    s = max(0.0, min(1.0, float(score)))

    # линейная интерполяция между start(BGR) и end(BGR)
    start = np.array([0, 0, 255], dtype=float)
    end   = np.array([36, 255, 12], dtype=float)
    color = (1.0 - s) * start + s * end
    return tuple(int(round(c)) for c in color)


def draw_box(
    frame: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    label: str,
    *,
    score: Optional[float] = None,
) -> None:
    """Нарисовать рамку и подпись, используя плавный цвет по score.

    Цвет рамки/подложки интерполируется от красного (score≈0) до зелёного (score≈1).

    Args:
        frame (np.ndarray): Кадр BGR (H×W×3, uint8), модифицируется in-place.
        xyxy (Tuple[int,int,int,int]): Координаты рамки (x1, y1, x2, y2).
        label (str): Текст подписи (например, "person 0.87").
        score (Optional[float]): Уверенность [0..1]; влияет на цвет.

    Returns:
        None
    """
    x1, y1, x2, y2 = xyxy
    color = score_to_color(score)

    # рамка
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # размеры текста
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

    # позиция подписи (над рамкой, но не выходим за верх)
    y_text = max(y1 - 6, th + 4)

    # подложка под текст (тем же цветом, чуть с отступами)
    cv2.rectangle(
        frame,
        (x1, y_text - th - 4),
        (x1 + tw + 6, y_text + base),
        color,
        -1,
    )

    # текст (чёрный для контраста)
    cv2.putText(
        frame,
        label,
        (x1 + 3, y_text - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )



def draw_pose(frame, kpts, skeleton=COCO_SKELETON, point_size=4, line_thickness=2):
    kpts = np.array(kpts)
    if kpts.ndim == 3:
        kpts = kpts[0]

    # линии
    for idx, (i, j) in enumerate(skeleton):
        if i >= len(kpts) or j >= len(kpts):
            continue

        x1, y1 = int(kpts[i][0]), int(kpts[i][1])
        x2, y2 = int(kpts[j][0]), int(kpts[j][1])

        if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
            continue

        color = POSE_COLORS[idx % len(POSE_COLORS)]
        cv2.line(frame, (x1, y1), (x2, y2), color, line_thickness)

    # точки
    for (x, y) in kpts[:, :2]:
        x, y = int(x), int(y)
        if x == 0 and y == 0:
            continue
        if x >= 0 and y >= 0:
            cv2.circle(frame, (x, y), point_size, (0, 255, 255), -1)


