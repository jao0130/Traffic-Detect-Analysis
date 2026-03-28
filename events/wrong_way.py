import numpy as np
from config import NORMAL_DIR, DIR_MIN_FRAMES, DIR_THRESHOLD, DIR_MOVE_MIN

_normal = np.array(NORMAL_DIR, dtype=float)
_normal /= np.linalg.norm(_normal)


def check(detections, state) -> list[dict]:
    events = []
    for det in detections:
        if len(state.history[det.track_id]) < DIR_MIN_FRAMES:
            continue
        direction = state.get_direction(det.track_id)
        if direction is None:
            continue
        move = np.array(direction, dtype=float)
        if np.linalg.norm(move) < DIR_MOVE_MIN:   # 靜止或位移不足，跳過
            continue
        if np.dot(move / np.linalg.norm(move), _normal) < -DIR_THRESHOLD:
            events.append({"event": "wrong_way", "track_id": det.track_id, "centroid": det.centroid})
    return events
