import json
import os
import time
from datetime import datetime

_output_file: str = "events_output/events.json"
_last_logged: dict[tuple, float] = {}

GRID_SIZE = 60   # 位置冷卻格大小（像素），同格內同類事件不重複記錄


def set_output(path: str):
    global _output_file, _last_logged
    _output_file = path
    _last_logged = {}


def _cooldown_key(event: dict) -> tuple:
    """以事件類型 + 位置格為冷卻 key，track_id 換了也不重複觸發。"""
    centroid = event.get("centroid")
    if centroid:
        gx, gy = centroid[0] // GRID_SIZE, centroid[1] // GRID_SIZE
        return (event["event"], gx, gy)
    return (event["event"], event["track_id"])


def log(event: dict, cooldown_sec: float):
    key = _cooldown_key(event)
    now = time.time()
    if now - _last_logged.get(key, 0) < cooldown_sec:
        return
    _last_logged[key] = now

    record = {**event, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    record.pop("centroid", None)
    print(f"[EVENT] {json.dumps(record, ensure_ascii=False)}")

    os.makedirs(os.path.dirname(_output_file), exist_ok=True)
    with open(_output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
