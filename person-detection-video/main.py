#!/usr/bin/env python3
"""Entry point: читать видео → детектить людей → рисовать → сохранять.

Поддерживает два режима:
  - обычная детекция людей (YOLOv8-detect);
  - детекция + pose-валидация и отрисовка поз (YOLOv8-pose).
"""

import argparse

from src.utils import open_video, read_video_props, create_writer
from src.infer import load_model, detect_persons, validate_with_pose
from src.drawing import draw_box, draw_pose
from src.drawing import draw_pose, COCO_SKELETON
from src.infer import compute_iou  



def parse_args() -> argparse.Namespace:
    """Принимает аргументы командной строки для скрипта детекции людей.

    Функция читает флаги, с которыми запущен скрипт, и возвращает объект
    с готовыми полями.
    Эти значения далее используются при чтении/записи видео и запуске модели.

    Доступные флаги:
      --input        Путь к входному видео (по умолчанию: crowd.mp4).
      --output       Путь к выходному видео с разметкой (out/crowd_annotated.mp4).
      --model        Весы модели YOLO для детекции (yolov8n.pt / yolov8s.pt и т.п.).
      --device       Устройство: auto / cuda / cpu (по умолчанию: auto).
      --conf         Порог уверенности детекций [0..1].
      --iou          Порог IoU для NMS [0..1].
      --stride       Обрабатывать каждый N-й кадр.
      --min_box      Отбрасывать слишком маленькие боксы (минимальная сторона, px).
      --imgsz        Входной размер для инференса (px).
      --max_det      Лимит детекций на кадр.

      --pose_validate  Включить валидацию детекций по YOLOv8-pose.
      --pose_model     Весы YOLOv8-pose для валидации и отрисовки поз.
    """
    p = argparse.ArgumentParser(
        description="Детекция людей на видео с опциональной pose-валидацией."
    )
    p.add_argument(
        "--input",
        type=str,
        default="crowd.mp4",
        help="Путь к входному видео.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="out/crowd_annotated.mp4",
        help="Путь к выходному MP4.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Весы YOLO для детекции (yolov8s.pt и т.п.).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Устройство инференса: auto/cuda/cpu.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Порог уверенности детекций.",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.55,
        help="Порог IoU для NMS.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Обрабатывать каждый N-й кадр для ускорения.",
    )
    p.add_argument(
        "--min_box",
        type=int,
        default=0.8,
        help="Отбрасывать боксы меньше указанной стороны (px).",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=1920,
        help="Входной размер для инференса YOLO (px).",
    )
    p.add_argument(
        "--max_det",
        type=int,
        default=1000,
        help="Лимит детекций на кадр.",
    )
    p.add_argument(
        "--pose_validate",
        action="store_true",
        help="Включить валидацию детекций по YOLOv8-pose и отрисовку поз.",
    )
    p.add_argument(
        "--pose_model",
        type=str,
        default="yolov8s-pose.pt",
        help="Весы YOLOv8-pose для валидации и отрисовки поз.",
    )

    return p.parse_args()


def main() -> None:
    """Главная точка входа: прочитать видео → детектировать людей → отрисовать → сохранить.

    Последовательность действий:
      1) Разобрать аргументы командной строки (`parse_args`).
      2) Открыть входное видео и считать его свойства (FPS, ширину, высоту).
      3) Создать видеописатель под выходной MP4 с тем же FPS и размером.
      4) Загрузить модель YOLO для детекции.
      5) При включённой опции `--pose_validate` загрузить YOLOv8-pose.
      6) Для каждого кадра:
         - при необходимости (по stride) выполнить детекцию людей;
         - опционально, посчитать позы YOLOv8-pose;
         - опционально, отфильтровать "сомнительные" детекции по позам;
         - при включённой pose-валидации — нарисовать скелеты людей;
         - нарисовать рамки и подписи `person {score:.2f}`;
         - записать кадр в выходное видео.
      7) Освободить ресурсы и вывести путь к результату.
    """
    args = parse_args()

    # 1. Открываем видео и читаем свойства
    cap = open_video(args.input)
    fps, w, h = read_video_props(cap)

    # 2. Создаём VideoWriter под выходной ролик
    writer = create_writer(args.output, fps, (w, h))

    # 3. Загружаем модель детекции
    model, device = load_model(args.model, args.device)

    # 4. При необходимости загружаем pose-модель
    pose_model = None
    if args.pose_validate:
        pose_model, _ = load_model(args.pose_model, args.device)

    # Основной цикл по кадрам
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % args.stride == 0:
            # --- Детекция людей ---
            dets = detect_persons(
                model=model,
                frame=frame,
                conf=args.conf,
                iou=args.iou,
                min_box=args.min_box,
                imgsz=args.imgsz,
                max_det=args.max_det,
            )

            pose_results = None

            if args.pose_validate and pose_model is not None:
                # --- Pose-инференс один раз на кадр ---
                pose_results = pose_model.predict(
                    source=frame,
                    conf=0.25,
                    verbose=False,
                )[0]

                # --- Pose-валидация детекций ---
                dets = validate_with_pose(pose_results, dets)

                # --- Отрисовка поз (скелетов) ---
                # if pose_results.keypoints is not None:
                #     for idx, person_kpts in enumerate(pose_results.keypoints.xy):
                #         conf_row = None
                #         if pose_results.keypoints.conf is not None:
                #             conf_row = pose_results.keypoints.conf[idx]
                #         draw_pose(frame, person_kpts, kpts_conf=conf_row)
                if pose_results.keypoints is not None and pose_results.boxes is not None:
                    pose_boxes = []
                    for i, b in enumerate(pose_results.boxes):
                        # на всякий случай фильтруем не-person (если у pose будет несколько классов)
                        cls_id = int(b.cls.item())
                        if cls_id != 0:  # COCO: 0 == person
                            continue

                        xyxy = tuple(map(int, b.xyxy[0].tolist()))  # (x1, y1, x2, y2)
                        pose_boxes.append((i, xyxy))   # сохраняем и индекс, и координаты

                    for det in dets:  # det — это (x1, y1, x2, y2, conf)
                        dx1, dy1, dx2, dy2, _ = det

                        best_iou = 0.0
                        best_idx = None

                        for i, (px1, py1, px2, py2) in pose_boxes:
                            iou = compute_iou((dx1, dy1, dx2, dy2), (px1, py1, px2, py2))
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = i

                        # если IoU достаточно большой — рисуем позу для ЭТОГО det
                        if best_idx is not None and best_iou >= 0.3:
                            kpts = pose_results.keypoints.xy[best_idx]
                            draw_pose(frame, kpts, COCO_SKELETON)


            # --- Отрисовка детекций (рамки + подписи) ---
            for x1, y1, x2, y2, conf in dets:
                draw_box(
                    frame,
                    (x1, y1, x2, y2),
                    f"person {conf:.2f}",
                    score=conf,
                )

        # Записываем кадр (даже если он не детектился из-за stride)
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"Сделано: {args.output}")


if __name__ == "__main__":
    main()
