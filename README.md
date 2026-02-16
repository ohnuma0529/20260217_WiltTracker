# WiltTracker-Training

葉の萎凋追跡モデル WiltTracker の訓練・評価パイプライン.

## ディレクトリ構造

```
WiltTracker-Training/
├── train.py              # 訓練スクリプト
├── eval.py               # 検証モード（推論 + 可視化）
├── configs/
│   ├── train.yaml        # 訓練設定
│   └── eval.yaml         # 検証設定
├── weights/
│   ├── YOLOv11_detect.pt # 葉検出モデル (YOLOv11)
│   └── WiltTracker.pt    # 追跡モデル (WiltTracker)
├── wilt_tracker/         # コアモジュール
│   ├── models/           # DecoupledTracker, Loss
│   ├── data/             # データセットクラス
│   └── utils/            # geometry, visualization, plots
├── runs/                 # 訓練結果
└── results/              # 検証結果
    └── {area}_{date}/
        ├── images/       # 可視化画像
        ├── video.mp4     # 可視化動画
        └── tracking.csv  # YOLO形式トラッキング結果
```

## 訓練データ

dataset_v2 (`/home/ohnuma/20251228_WiltTracker/dataset_v2`) を使用.

### 対象区画・日付

| 区画 | 訓練日付 | 検証日付 |
|------|----------|----------|
| 31 | 20250430, 20250514, 20250524, 20250607, 20250616, 20251024, 20251127 | 20251114 |
| 32 | 20250501, 20250512, 20250524, 20250531, 20250613, 20250715, 20250725, 20250817, 20251001, 20251024, 20251108 | — |
| 34 | 20250430, 20250510, 20250518, 20250530, 20250607, 20250617, 20250719, 20250728, 20250810, 20250815, 20250824, 20251024, 20251030 | 20251114 |

> 日付に `20251114` を含むデータが検証用，それ以外が全て訓練用として使用される.

## 使い方

### 訓練

```bash
python train.py --config configs/train.yaml
```

### 検証モード

```bash
python eval.py --config configs/eval.yaml
```

## コンフィグファイル

### `configs/eval.yaml` — 検証設定

```yaml
detection:                           # YOLOv11 による葉の初期検出
  weights: weights/YOLOv11_detect.pt # 検出モデルの重み
  conf_threshold: 0.3                # 検出信頼度の閾値
  top_n: 5                           # 追跡対象の葉の最大数（面積上位）

tracking:                            # WiltTracker による自己回帰追跡
  weights: weights/WiltTracker.pt    # 追跡モデルの重み
  backbone: densenet121              # バックボーン構造（ロード時に必要）
  multi_head: true                   # マルチヘッド出力
  past_frames: 5                     # 参照する過去フレーム数
  past_stride: 5                     # 過去フレームの間隔（分単位）
  use_mixed_stride: true             # 不規則な時間間隔を使用

input:                               # 入力設定
  image_dir: /path/to/images         # 画像ディレクトリのパス
  time_range: ["07:00", "17:00"]     # 追跡対象の時間帯

output:                              # 出力設定
  dir: results                       # 出力ルートディレクトリ
  save_images: true                  # 可視化画像の保存
  save_video: true                   # 可視化動画 (video.mp4) の保存
  save_csv: true                     # YOLO形式 CSV (tracking.csv) の保存
```

### `configs/train.yaml` — 訓練設定

| セクション | 主な項目 | 説明 |
|-----------|---------|------|
| `data` | `dataset_dir`, `img_size`, `past_frames`, `past_stride` | データセットパス，入力サイズ，過去フレーム参照の設定 |
| `model` | `backbone`, `multi_head`, `dropout` | モデル構造の指定 |
| `training` | `epochs`, `batch_size`, `learning_rate`, `patience` | 学習ハイパーパラメータ |
| `training.stages` | `w_attn`, `w_consistency`, `reward_*` | カリキュラム学習のステージ設定 |
| `experiment` | `name`, `seed` | 実験名とランダムシード |

## 出力形式

### tracking.csv

YOLO形式の正規化座標 (0-1) で記録される.

| 列名 | 説明 |
|------|------|
| `frame` | 画像ファイル名 |
| `leaf_id` | 葉の ID |
| `cx`, `cy` | BBox 中心座標（正規化） |
| `w`, `h` | BBox 幅・高さ（正規化） |
| `base_x`, `base_y` | 葉の基部座標（正規化） |
| `tip_x`, `tip_y` | 葉の先端座標（正規化） |

この CSV と元画像があれば可視化画像を再現可能.
