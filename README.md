# Traffic Event Analysis Module

簡化版即時交通事件分析模組，利用 YOLO11 物件辨識結果，判斷是否發生特定交通違規事件。

---

## 環境需求

| 項目 | 規格 |
|------|------|
| Python | 3.11+ |
| 套件管理 | Anaconda |
| YOLO 版本 | YOLO11 |
| 推論裝置 | CUDA（自動偵測，無 GPU 則 fallback 至 CPU）|

```bash
conda create -n traffic python=3.11
conda activate traffic
pip install -r requirements.txt
```

---

## 硬體環境

| 項目 | 規格 |
|------|------|
| CPU | Intel Core i5-12400F（6C/12T） |
| RAM | 32GB DDR4-3200 |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| VRAM | 16GB |

### GPU 使用率

推論使用 `yolo11n.pt`（nano 模型），RTX 4060 Ti 上平均 GPU 使用率約 **20–35%**，VRAM 佔用約 **1.5–2GB**。

---

## 模型來源

使用 Ultralytics 官方預訓練模型，無需自行訓練：

```bash
# 首次執行時自動下載，或手動放置於 models/
# https://github.com/ultralytics/assets/releases
yolo11n.pt   # nano，速度優先
```

偵測類別：`car`, `truck`, `bus`, `motorcycle`, `person`

---

## 執行方式

```bash
python main.py --video data/車輛逆向.mp4
python main.py --video data/車輛停等.mp4
```

啟動後會先跳出 ROI 選取視窗，框選禁行區域後按 Enter 開始偵測。

### 快捷鍵

| 鍵 | 功能 |
|----|------|
| `Q` | 結束並輸出影片 |
| `R` | 重新選取 ROI |
| `P` / `Space` | 暫停 / 繼續 |

---

## 事件類型

同時偵測三種事件，結果輸出至畫面、console 及 JSON 檔。

### 逆向行駛（wrong_way）
- 計算車輛 centroid 歷史軌跡的移動向量
- 與設定的正常行進方向做 dot product
- 向量長度 ≥ `DIR_MOVE_MIN`，且 dot product < `-DIR_THRESHOLD` → 觸發

### 禁行區域進入（forbidden_zone）
- 偵測車輛 centroid 進入 ROI 多邊形邊界的瞬間觸發
- ROI 可在啟動時互動設定，或按 `R` 重新設定，支援設為空（停用）

### 停留過久（overstay）
- 移動向量長度 < `OVERSTAY_MOVE_THRESHOLD` 判定為靜止
- 連續靜止超過 `OVERSTAY_SECONDS` 秒 → 觸發
- 需連續移動 `OVERSTAY_RESET_FRAMES` 幀才重置計時（防短暫減速誤判）

---

## 輸出

```
events_output/
├── output_車輛逆向.mp4     # 標注後影片
├── events_車輛逆向.json    # 事件 log
├── output_車輛停等.mp4
└── events_車輛停等.json
```

JSON 格式範例：

```json
{"event": "wrong_way", "track_id": 5, "time": "2026-01-05 14:32:10"}
{"event": "forbidden_zone", "track_id": 3, "time": "2026-01-05 14:32:15"}
{"event": "overstay", "track_id": 7, "duration": 8.3, "time": "2026-01-05 14:32:20"}
```

---

## 分析速度（FPS）

| 模式 | FPS |
|------|-----|
| GPU（RTX 4060 Ti）| 55–90 FPS |
| CPU only | 8–15 FPS |

---

## 專案結構

```
├── main.py              # 入口（ROI 設定 → 偵測 → 輸出影片）
├── config.py            # 所有參數集中管理
├── core/
│   ├── detector.py      # YOLO11 推論（含邊緣 padding）
│   └── tracker_state.py # 追蹤狀態、ghost 機制、位移跳躍偵測
├── events/
│   ├── wrong_way.py     # 逆向偵測
│   ├── forbidden_zone.py# 禁行區偵測
│   └── overstay.py      # 停留過久偵測
├── output/
│   ├── visualizer.py    # cv2 繪圖
│   └── event_logger.py  # JSON + console log
└── tools/
    └── roi_selector.py  # 互動式 ROI 選取工具
```

---

## 開發中遇到的問題

1. **遮蓋導致 track_id 重置**：被其他車遮擋後，YOLO tracker 重新分配新 ID，overstay 計時歸零。
   → 實作 ghost 機制，車輛消失時暫存位置與計時，新 ID 出現在附近自動繼承。

2. **靜止車 ID 被經過車輛繼承**：ByteTrack 在邊緣區域將靜止車的 ID 轉給路過車輛，導致計時被錯誤重置。
   → 記錄最後靜止 centroid，`moving_streak` 達門檻時在該位置存 ghost，讓靜止車新 ID 繼承。

3. **邊緣偵測不穩定**：部分遮擋的車輛 bbox 不完整，造成 tracker 頻繁掉幀。
   → 推論前對 frame 加 `BORDER_REFLECT` padding，讓邊緣物件有完整上下文。

4. **逆向偵測誤判靜止車**：近零向量因偵測抖動方向隨機，被誤判為逆向。
   → 加入 `DIR_MOVE_MIN` 最小位移門檻，靜止狀態完全跳過方向判斷。

5. **短暫減速被誤判為停留**：慢速通過的車輛有幾幀向量偏小，誤觸發 overstay。
   → 加長歷史窗口（`TRACKER_HISTORY_LEN=40`）並提高 `OVERSTAY_RESET_FRAMES`。

---

## 開發總時長

約 **6–8 小時**

---
