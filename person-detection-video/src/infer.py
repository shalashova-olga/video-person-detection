from typing import List, Tuple, Optional
import numpy as np
from ultralytics import YOLO
import numpy as np


def load_model(weights: str, device: Optional[str]):
    """Загрузить модель YOLO и определить устройство инференса.

    Args:
        weights: Путь или имя весов модели YOLO (yolov8s.pt, yolov8s-pose.pt и т.п.).
        device: Предпочитаемое устройство:
            - "auto" или None — выбор оставляем Ultralytics;
            - "cuda" — явно GPU;
            - "cpu" — явно CPU.

    Returns:
        (model, chosen_device):
            - model: загруженная модель YOLO.
            - chosen_device: строка устройства или None.
    """
    chosen_device = None if device in ("auto", None) else device
    model = YOLO(weights)
    return model, chosen_device


def detect_persons(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    iou: float,
    min_box: int,
    imgsz: int,
    max_det: Optional[int] = None,
) -> List[Tuple[int, int, int, int, float]]:
    """Выполнить детекцию людей на одном кадре и вернуть боксы с уверенностью.

    Делает predict, фильтрует по классу `person` (id=0 в COCO) и отбрасывает
    слишком маленькие боксы.

    Args:
        model: Загруженная модель YOLO (detect).
        frame: Кадр BGR (H×W×3, uint8).
        conf: Порог уверенности [0..1].
        iou: Порог IoU для NMS [0..1].
        min_box: Минимальный размер стороны бокса (px).
        imgsz: Входной размер для инференса (px).
        max_det: Максимальное число детекций на кадр.

    Returns:
        Список (x1, y1, x2, y2, conf) только для класса person.
    """
    predict_kwargs = dict(
        source=frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    if max_det is not None:
        predict_kwargs["max_det"] = max_det

    results = model.predict(**predict_kwargs)
    r = results[0]
    out: List[Tuple[int, int, int, int, float]] = []

    if r.boxes is None or len(r.boxes) == 0:
        return out

    for b in r.boxes:
        # COCO: class id 0 == person
        cls_id = int(b.cls.item())
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

        # фильтр по размеру бокса
        if min(x2 - x1, y2 - y1) < min_box:
            continue

        conf_score = float(b.conf.item())
        out.append((x1, y1, x2, y2, conf_score))

    return out


def validate_with_pose(
    pose_results,
    dets: List[Tuple[int, int, int, int, float]],
    iou_thresh: float = 0.3,
    min_kpts: int = 2,
    kpt_conf: float = 0.2,
    skip_conf: float = 0.6,
) -> List[Tuple[int, int, int, int, float]]:
    """Отфильтровать детекции по уже посчитанным pose-результатам.

    Идея:
      - уверенные детекции (conf >= skip_conf) не трогаем;
      - слабые боксы оставляем только если рядом есть pose-бокс класса person
        с достаточным числом уверенных ключевых точек.

    Args:
        pose_results: Результат pose_model.predict(...)[0].
        dets: Детекции (x1, y1, x2, y2, conf) от детектора.
        iou_thresh: Минимальный IoU между детектор-боксом и pose-боксом.
        min_kpts: Минимальное число "надёжных" ключевых точек.
        kpt_conf: Порог уверенности ключевых точек.
        skip_conf: Детекции с conf >= skip_conf не фильтруем.

    Returns:
        Отфильтрованный список детекций.
    """
    if pose_results is None or not dets:
        return dets

    boxes = pose_results.boxes
    kpts_obj = getattr(pose_results, "keypoints", None)

    if boxes is None or kpts_obj is None or len(boxes) == 0:
        # pose ничего полезного не дала — не фильтруем
        return dets

    kpts_conf_all = getattr(kpts_obj, "conf", None)
    if kpts_conf_all is None:
        # нет уверенностей по точкам — тоже не фильтруем
        return dets

    kpts_conf_all = np.array(kpts_conf_all)  # shape (N, K)

    pose_boxes: List[Tuple[Tuple[int, int, int, int], int]] = []

    # Собираем боксы pose-модели + количество "хороших" ключевых точек
    for i, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        # COCO: person == 0
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        conf_row = np.array(kpts_conf_all[i]).flatten()
        good_kpts = int((conf_row > kpt_conf).sum())

        pose_boxes.append(((x1, y1, x2, y2), good_kpts))

    if not pose_boxes:
        # pose не нашла ни одного person — не трогаем детекции
        return dets

    def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by1) if False else min(ay2, by2)  # just safety
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    filtered: List[Tuple[int, int, int, int, float]] = []

    for det in dets:
        x1, y1, x2, y2, conf = det

        # Уверенные детекции не трогаем
        if conf >= skip_conf:
            filtered.append(det)
            continue

        best_iou = 0.0
        best_kpts = 0

        for (px1, py1, px2, py2), kcnt in pose_boxes:
            i = iou((x1, y1, x2, y2), (px1, py1, px2, py2))
            if i > best_iou:
                best_iou = i
                best_kpts = kcnt

        # слабый бокс оставляем только если есть "скелет" рядом
        if best_iou >= iou_thresh and best_kpts >= min_kpts:
            filtered.append(det)

    return filtered




def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    return inter / (area_a + area_b - inter)