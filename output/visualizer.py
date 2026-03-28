import cv2
import numpy as np
import config

_EVENT_COLOR = {
    "wrong_way":      (0, 0, 255),
    "forbidden_zone": (0, 140, 255),
    "overstay":       (255, 50, 50),
}
_SHORTCUTS = "[Q]Quit  [R]Re-ROI  [P]Pause"


def draw(frame, vehicles, persons, events, paused=False):
    # ROI（空列表則不畫）
    if config.ROI_FORBIDDEN:
        cv2.polylines(frame, [np.array(config.ROI_FORBIDDEN)], True, (0, 255, 0), 2)

    # 車輛 bbox
    for det in vehicles:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(frame, f"{det.class_name} #{det.track_id}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # 行人 bbox
    for det in persons:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 1)
        cv2.putText(frame, f"person #{det.track_id}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

    # 事件標注
    for i, ev in enumerate(events):
        color = _EVENT_COLOR.get(ev["event"], (0, 255, 255))
        cx, cy = ev["centroid"]
        label = f"[{ev['event'].upper()}] ID:{ev['track_id']}"
        if "duration" in ev:
            label += f" {ev['duration']}s"
        cv2.putText(frame, label, (cx - 60, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, label, (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # 快捷鍵提示（右下角）
    h, w = frame.shape[:2]
    cv2.putText(frame, _SHORTCUTS, (w - 320, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # 暫停提示
    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    return frame
