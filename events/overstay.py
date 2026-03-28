import numpy as np
from config import OVERSTAY_SECONDS, OVERSTAY_MOVE_THRESHOLD, DIR_MIN_FRAMES, OVERSTAY_RESET_FRAMES

_moving_streak:      dict[int, int]   = {}  # track_id -> 連續非靜止幀數
_last_stationary_pos: dict[int, tuple] = {}  # track_id -> 最後一次靜止的 centroid


def _is_stationary(state, track_id) -> bool:
    if len(state.history[track_id]) < DIR_MIN_FRAMES:
        return False
    direction = state.get_direction(track_id)
    if direction is None:
        return False
    return np.linalg.norm(direction) < OVERSTAY_MOVE_THRESHOLD


def check(detections, state) -> list[dict]:
    events = []
    active_ids = {det.track_id for det in detections}

    # 消失的車輛清除計時與紀錄
    for tid in list(state.entry_time):
        if tid not in active_ids:
            state.mark_left(tid)
            _moving_streak.pop(tid, None)
            _last_stationary_pos.pop(tid, None)

    for det in detections:
        if _is_stationary(state, det.track_id):
            _moving_streak[det.track_id] = 0
            _last_stationary_pos[det.track_id] = det.centroid   # 記錄靜止位置
            state.mark_entered(det.track_id, det.centroid)
            elapsed = state.time_in_roi(det.track_id)
            if elapsed >= OVERSTAY_SECONDS:
                events.append({
                    "event": "overstay",
                    "track_id": det.track_id,
                    "centroid": det.centroid,
                    "duration": round(elapsed, 1),
                })
        else:
            streak = _moving_streak.get(det.track_id, 0) + 1
            _moving_streak[det.track_id] = streak
            if streak >= OVERSTAY_RESET_FRAMES:
                # ID 被移動中的車繼承 → 在最後靜止位置存 ghost，讓原本靜止車的新 ID 繼承
                state.mark_left(det.track_id,
                                 ghost_centroid=_last_stationary_pos.get(det.track_id))
                _moving_streak[det.track_id] = 0
                _last_stationary_pos.pop(det.track_id, None)

    return events
