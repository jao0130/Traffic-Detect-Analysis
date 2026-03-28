import argparse
import re
import time
import os
import cv2

import config
from core.detector import Detector
from core.tracker_state import TrackerState
from events import wrong_way, forbidden_zone, overstay
from output.visualizer import draw
from output.event_logger import log
from tools.roi_selector import select_roi


def _save_roi_to_config(points: list[tuple]):
    with open("config.py", "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(
        r"ROI_FORBIDDEN\s*=\s*\[.*?\]",
        f"ROI_FORBIDDEN = {points}",
        content,
    )
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(content)


def _apply_roi(roi: list[tuple] | None):
    """roi=None 表示取消，roi=[] 表示清除，否則套用新 ROI。"""
    if roi is None:
        return
    config.ROI_FORBIDDEN = roi
    _save_roi_to_config(roi)
    status = "已清除" if roi == [] else f"已更新：{roi}"
    print(f"ROI {status}")


def main():
    parser = argparse.ArgumentParser(description="Traffic Event Analyzer")
    parser.add_argument("--video", required=True, help="影片路徑（mp4）")
    args = parser.parse_args()

    # --- 初始 ROI 設定 ---
    print("請框選禁行區域 ROI（C=清除/不設定，Q=保留現有）")
    _apply_roi(select_roi(args.video))

    detector = Detector(config.MODEL_PATH, config.TARGET_CLASSES, config.DEVICE)
    state    = TrackerState(history_len=config.TRACKER_HISTORY_LEN)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片：{args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("events_output", exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    out_path   = f"events_output/output_{video_stem}.mp4"
    writer     = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (width, height))

    from output import event_logger
    event_logger.set_output(f"events_output/events_{video_stem}.json")

    frame_idx = 0
    prev_time = time.time()
    paused    = False
    last_frame = None
    prev_ids  = set()

    while cap.isOpened():
        key = cv2.waitKey(1) & 0xFF

        # --- 快捷鍵 ---
        if key == ord("q"):
            break
        elif key == ord("p") or key == 32:   # P 或 Space
            paused = not paused
        elif key == ord("r"):                 # R = 重新選 ROI
            cv2.destroyAllWindows()
            _apply_roi(select_roi(args.video))

        if paused:
            if last_frame is not None:
                cv2.imshow("Traffic Analysis", draw(last_frame.copy(), [], [], [], paused=True))
            continue

        ret, frame = cap.read()
        if not ret:
            break

        detections  = detector.run(frame)
        current_ids = {d.track_id for d in detections}
        vehicles    = [d for d in detections if d.class_name in config.VEHICLE_CLASSES]
        persons     = [d for d in detections if d.class_name == "person"]

        for det in detections:
            state.update(det.track_id, det.centroid)

        # 即時建立 ghost（車輛剛消失這幀）
        lost = prev_ids - current_ids
        if lost:
            state.on_lost(lost)
        prev_ids = current_ids

        if frame_idx % 30 == 0:
            state.cleanup(current_ids)

        events = (
            wrong_way.check(vehicles, state) +
            forbidden_zone.check(vehicles, state) +
            overstay.check(vehicles, state)
        )
        for ev in events:
            log(ev, config.LOG_COOLDOWN_SEC)

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        frame = draw(frame, vehicles, persons, events)
        cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        last_frame = frame.copy()
        writer.write(frame)
        cv2.imshow("Traffic Analysis", frame)
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"完成。輸出影片：{out_path}")
    print("事件記錄：events_output/events.json")


if __name__ == "__main__":
    main()
