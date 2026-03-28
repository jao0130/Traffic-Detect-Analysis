import time
import numpy as np
from collections import defaultdict, deque
import config


class TrackerState:
    def __init__(self, history_len=40):
        self.history     = defaultdict(lambda: deque(maxlen=history_len))
        self.entry_time  = {}   # track_id -> 開始靜止的時間（overstay）
        self.in_roi_prev = {}   # track_id -> 上一幀是否在 ROI（forbidden zone）
        self._ghosts     = {}   # centroid_tuple -> (entry_time, ghost_timestamp)

    def update(self, track_id, centroid):
        history = self.history[track_id]
        if history:
            jump = np.linalg.norm(np.array(centroid) - np.array(history[-1]))
            if jump > config.MAX_JUMP_DIST:
                # 位移不合理 → tracker 把 ID 誤給另一輛車
                # 先把舊位置存成 ghost，再重置此 ID 的所有狀態
                if track_id in self.entry_time:
                    self._ghosts[tuple(history[-1])] = (self.entry_time[track_id], time.time())
                history.clear()
                self.entry_time.pop(track_id, None)
                self.in_roi_prev.pop(track_id, None)
        history.append(centroid)

    def get_direction(self, track_id):
        pts = self.history[track_id]
        if len(pts) < 2:
            return None
        return (pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])

    def mark_entered(self, track_id, centroid):
        if track_id not in self.entry_time:
            inherited = self._inherit_ghost(centroid)
            self.entry_time[track_id] = inherited if inherited else time.time()

    def mark_left(self, track_id, ghost_centroid=None):
        """清除計時；若提供 ghost_centroid，先把舊計時存成 ghost 供新 ID 繼承。"""
        if ghost_centroid and track_id in self.entry_time:
            self._ghosts[tuple(ghost_centroid)] = (self.entry_time[track_id], time.time())
        self.entry_time.pop(track_id, None)

    def time_in_roi(self, track_id) -> float:
        if track_id not in self.entry_time:
            return 0.0
        return time.time() - self.entry_time[track_id]

    def on_lost(self, lost_ids: set):
        """車輛剛消失時立即建立 ghost（每幀呼叫）。"""
        now = time.time()
        for tid in lost_ids:
            if tid in self.entry_time and self.history[tid]:
                centroid = tuple(self.history[tid][-1])
                self._ghosts[centroid] = (self.entry_time[tid], now)

    def cleanup(self, active_ids: set):
        now = time.time()
        stale = set(self.history) - active_ids
        for tid in stale:
            # 車輛消失時，若正在靜止計時則存為 ghost
            if tid in self.entry_time and self.history[tid]:
                centroid = tuple(self.history[tid][-1])
                self._ghosts[centroid] = (self.entry_time[tid], now)
            del self.history[tid]
            self.entry_time.pop(tid, None)
            self.in_roi_prev.pop(tid, None)

        # 過期 ghost
        self._ghosts = {
            k: v for k, v in self._ghosts.items()
            if now - v[1] < config.GHOST_TTL
        }

    def _inherit_ghost(self, centroid) -> float | None:
        best_dist, best_key = config.GHOST_DIST_THRESHOLD, None
        for ghost_centroid in self._ghosts:
            dist = np.linalg.norm(np.array(centroid) - np.array(ghost_centroid))
            if dist < best_dist:
                best_dist, best_key = dist, ghost_centroid
        if best_key:
            entry_time = self._ghosts.pop(best_key)[0]
            return entry_time
        return None
