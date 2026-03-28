import numpy as np
import cv2
import config


def _in_roi(point) -> bool:
    if not config.ROI_FORBIDDEN:
        return False
    roi = np.array(config.ROI_FORBIDDEN)
    return cv2.pointPolygonTest(roi, point, False) >= 0


def check(detections, state) -> list[dict]:
    if not config.ROI_FORBIDDEN:
        return []

    events = []
    for det in detections:
        now_in  = _in_roi(det.centroid)
        prev_in = state.in_roi_prev.get(det.track_id, False)

        if now_in and not prev_in:
            events.append({"event": "forbidden_zone", "track_id": det.track_id, "centroid": det.centroid})

        state.in_roi_prev[det.track_id] = now_in
    return events
