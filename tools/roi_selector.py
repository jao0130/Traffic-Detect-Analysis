"""
互動式 ROI 選取工具。
  Left click  = 新增頂點
  Enter       = 確認（至少 3 點）
  C           = 清除 ROI（回傳空列表，停用禁行區偵測）
  R           = 重設點位
  Q           = 取消（保留現有設定）
"""
import argparse
import cv2
import numpy as np


def select_roi(video_path: str) -> list[tuple] | None:
    """
    Returns:
      list of tuples  → 新 ROI
      []              → 清除 ROI（按 C）
      None            → 取消，保留現有設定（按 Q）
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise FileNotFoundError(f"無法讀取：{video_path}")

    points = []
    result = None   # None = cancelled

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    win = "ROI Selector [Click=add  Enter=confirm  C=clear  R=reset  Q=cancel]"
    cv2.namedWindow(win)
    cv2.imshow(win, frame)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        display = frame.copy()
        for p in points:
            cv2.circle(display, p, 5, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.polylines(display, [np.array(points)], True, (0, 255, 0), 2)

        hint = "Enter=confirm" if len(points) >= 3 else f"need {3 - len(points)} more points"
        cv2.putText(display, hint, (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(win, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(points) >= 3:   # Enter
            result = points[:]
            break
        elif key == ord("c"):                 # C = 清除 ROI
            result = []
            break
        elif key == ord("r"):                 # R = 重設點位
            points.clear()
        elif key == ord("q"):                 # Q = 取消
            result = None
            break

    cv2.destroyAllWindows()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    roi = select_roi(args.video)
    if roi is None:
        print("取消，保留現有設定。")
    elif roi == []:
        print("ROI 已清除。")
    else:
        print(f"\nROI = {roi}")
