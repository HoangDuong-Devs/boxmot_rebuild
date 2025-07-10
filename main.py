import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS
import traceback

# Cấu hình tracker
TRACKER_TYPE   = "bytetrack" 
tracker_config = TRACKER_CONFIGS / f"{TRACKER_TYPE}.yaml" 

tracker = create_tracker(
    tracker_type=TRACKER_TYPE,
    tracker_config=tracker_config,
    half=False,
    per_class=False
)

# Thêm thư mục gốc yolov7 vào sys.path để "utils.general" hoạt động
YOLOV7_PATH = os.path.join(os.path.dirname(__file__), "yolov7")
sys.path.insert(0, YOLOV7_PATH)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from yolov7.utils.datasets      import letterbox
from yolov7.utils.torch_utils   import select_device
from yolov7.models.experimental import attempt_load
from yolov7.utils.general       import non_max_suppression, scale_coords

@dataclass
class Config:
    """Configuration class"""
    IMAGE_PATH          : str   = r"test_image.jpg"
    WEIGHTS_PATH        : str   = r"best.pt"
    DEVICE              : str   = '0'
    CONF_THRESHOLD      : float = 0.20
    IOU_THRESHOLD       : float = 0.45
    PERSON_CLASS_ID     : int   = 2
    OUTPUT_IMAGE        : str   = "output_image.jpg"  # Tên file xuất
    MAX_IMAGE_WIDTH     : int   = 1280

@dataclass
class DetectionResult:
    """Class to store detection results"""
    id: int
    bbox: Tuple[int, int, int, int]
    confidence: float = 0.0

config = Config()
device = select_device(config.DEVICE)
model  = attempt_load(config.WEIGHTS_PATH, map_location=device)
model.eval()
logger.info(f"✅ YOLO model loaded successfully on device: {device}")

frame = cv2.imread(config.IMAGE_PATH)
if frame is None:
    logger.error("❌ Không đọc được ảnh!")
    sys.exit(1)

original_frame = frame.copy()
# Giữ bản gốc dùng cho tracker
try:
    # Preprocess image
    img = letterbox(frame, new_shape=640)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0

    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        pred = model(img_tensor, augment=False)[0]
        detections = non_max_suppression(
            pred,
            conf_thres=config.CONF_THRESHOLD,
            iou_thres=config.IOU_THRESHOLD
        )

    # Bảng màu nhiều màu sẵn
    colors = [
        (255, 0, 0),    # đỏ
        (0, 255, 0),    # xanh lá
        (0, 0, 255),    # xanh dương
        (255, 255, 0),  # vàng
        (255, 0, 255),  # hồng
        (0, 255, 255),  # cyan
        (128, 0, 128),  # tím
        (255, 127, 0),  # cam
        (0, 128, 255),  # xanh da trời
        (128, 255, 0)   # xanh non
    ]

    persons = []
    person_id = 1  # ID riêng cho người

    for det in detections:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()

            # Lấy các bbox người
            xywhs = []
            confs = []
            clss = []

            for (*xyxy, conf, cls) in det:
                if int(cls.item()) == config.PERSON_CLASS_ID:
                    x1, y1, x2, y2 = map(int, xyxy)
                    w  = x2 - x1
                    h  = y2 - y1
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    xywhs.append([cx, cy, w, h])
                    confs.append(conf.item())
                    clss.append(cls.item())

            # Convert về np.array
            xywhs = np.array(xywhs)
            confs = np.array(confs)
            clss  = np.array(clss)

            # ⚠️ Bắt buộc có frame hiện tại
            outputs = tracker.update(xywhs, confs, clss, original_frame)

            if len(outputs):
                
                if isinstance(outputs, np.ndarray):
                    print(">>> DEBUG: outputs dtype:", outputs.dtype)

                for x1, y1, x2, y2, track_id, cls in outputs:
                    x1, y1, x2, y2, track_id = map(int, (x1, y1, x2, y2, track_id))
                    box_color = colors[(track_id - 1) % len(colors)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    persons.append(DetectionResult(id=track_id, bbox=(x1, y1, x2, y2)))

    logger.info(f"✅ Detected {len(persons)} persons")

    # Xuất ảnh đã vẽ ra file
    success = cv2.imwrite(config.OUTPUT_IMAGE, frame)
    if success:
        logger.info(f"💾 Saved output image: {config.OUTPUT_IMAGE}")
    else:
        logger.error(f"❌ Failed to save output image!")

except Exception as e:
    logger.error(f"❌ Error in person detection: {e}")
    traceback.print_exc()
