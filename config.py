import torch

MODEL_PATH = "models/yolo11n.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_CLASSES   = {"car", "truck", "bus", "motorcycle", "person"}
VEHICLE_CLASSES  = {"car", "truck", "bus", "motorcycle"}  # 事件偵測範圍
DETECT_PAD = 40
# --- ROI 設定（使用 tools/roi_selector.py 點選後貼上座標）---
ROI_FORBIDDEN = [(555, 94), (446, 234), (436, 231)]

# 追蹤歷史窗口（幀數），影響靜止/方向判斷的穩定性
TRACKER_HISTORY_LEN = 40

# 邊緣偵測補強：推論前對 frame 加 padding，讓邊緣物件有足夠上下文\nDETECT_PAD = 40

# 逆向偵測
NORMAL_DIR     = (1, 0)   # 正常行進方向向量 (dx, dy)
DIR_MIN_FRAMES = 25       # 至少累積幾幀才計算方向
DIR_THRESHOLD  = 0.6      # dot product < -threshold 視為逆向
DIR_MOVE_MIN   = 30       # 最小位移（像素），低於此視為靜止，不做方向判斷

# 停留過久
OVERSTAY_SECONDS        = 5    # 靜止超過幾秒觸發
OVERSTAY_MOVE_THRESHOLD = 10   # 移動向量長度 < 此值視為靜止（像素）
OVERSTAY_RESET_FRAMES   = 20   # 需連續移動幾幀才重置計時（避免短暫減速誤判）

# 遮蓋補償：track_id 消失後保留 ghost，新 ID 在附近則繼承計時
GHOST_TTL            = 3    # ghost 保留秒數
GHOST_DIST_THRESHOLD = 80   # 繼承距離上限（像素）

# 事件 log 冷卻
LOG_COOLDOWN_SEC = 3

# tracker ID 誤判保護：同一 ID 兩奨間位移超過此値（像素）則視為誤判並重置
MAX_JUMP_DIST = 120
