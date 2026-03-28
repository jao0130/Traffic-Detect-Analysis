import cv2
from dataclasses import dataclass
from ultralytics import YOLO
import config


@dataclass
class Detection:
    track_id: int
    class_name: str
    bbox: tuple       # (x1, y1, x2, y2)，已還原為原始座標
    confidence: float
    centroid: tuple   # (cx, cy)


class Detector:
    def __init__(self, model_path, target_classes, device):
        self.model = YOLO(model_path)
        self.target_classes = target_classes
        self.device = device

    def run(self, frame) -> list[Detection]:
        pad = config.DETECT_PAD
        h, w = frame.shape[:2]

        # 加 padding 讓邊緣物件有足夠上下文
        padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        results = self.model.track(padded, persist=True, device=self.device,
                                   verbose=False, conf=0.3)

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if box.id is None:
                    continue
                cls_name = self.model.names[int(box.cls)]
                if cls_name not in self.target_classes:
                    continue

                # 還原座標（減去 padding）並 clamp 到原始畫面範圍
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 - pad)
                y2 = min(h, y2 - pad)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                detections.append(Detection(
                    track_id=int(box.id),
                    class_name=cls_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf),
                    centroid=(cx, cy),
                ))
        return detections
