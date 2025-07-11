import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS

# Add YOLOv7 repo to path so its utils can be imported
YOLOV7_ROOT = Path(__file__).parent / "yolov7"
sys.path.insert(0, str(YOLOV7_ROOT))

from yolov7.utils.datasets import letterbox
from yolov7.utils.torch_utils import select_device
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords


@dataclass
class TrackingConfig:
    """Simple configuration for YOLOv7 based tracking."""

    source: str = "data/videos/demo.mp4"  # video or camera source
    weights: str = "best.pt"  # path to YOLOv7 weights
    tracker_type: str = "bytetrack"  # tracker name
    device: str = "0"  # cuda device or cpu
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    output: str = "tracked_output.avi"
    img_size: int = 640
    half: bool = False
    per_class: bool = False


def run_tracking(cfg: TrackingConfig) -> None:
    """Run detection with YOLOv7 and update the selected tracker."""

    device = select_device(cfg.device)
    model = attempt_load(cfg.weights, map_location=device)
    model.eval()

    tracker = create_tracker(
        tracker_type=cfg.tracker_type,
        tracker_config=TRACKER_CONFIGS / f"{cfg.tracker_type}.yaml",
        half=cfg.half,
        per_class=cfg.per_class,
    )

    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {cfg.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        cfg.output, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)
    )

    colors: List[tuple] = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = letterbox(frame, new_shape=cfg.img_size)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor, augment=False)[0]
            det = non_max_suppression(
                pred, conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres
            )[0]

        if det is not None and len(det):
            det[:, :4] = scale_coords(
                img_tensor.shape[2:], det[:, :4], frame.shape
            ).round()
            xywhs = []
            confs = []
            clss = []
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                w = x2 - x1
                h = y2 - y1
                xywhs.append([x1 + w / 2, y1 + h / 2, w, h])
                confs.append(conf.item())
                clss.append(int(cls.item()))

            outputs = tracker.update(
                np.asarray(xywhs),
                np.asarray(confs),
                np.asarray(clss),
                frame,
            )

            for x1, y1, x2, y2, tid, cls in outputs:
                x1, y1, x2, y2, tid = map(int, (x1, y1, x2, y2, tid))
                color = colors[tid % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {tid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        out.write(frame)

    cap.release()
    out.release()
    print(f"Tracking finished. Results saved to {cfg.output}")


if __name__ == "__main__":
    config = TrackingConfig()
    run_tracking(config)
