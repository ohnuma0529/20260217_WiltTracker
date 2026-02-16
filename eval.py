"""
WiltTracker 検証モード (Evaluation)
正解ラベルのない画像群に対し，YOLOv11 による初期検出と
WiltTracker による自己回帰追跡を行うパイプライン．

各日 (7:00-17:00) を独立して追跡し，元画像上に全追跡葉の
BBox, Keypoint, ID を描画して保存する．

出力:
  - 可視化画像 (images/)
  - 可視化動画 (video.mp4)
  - トラッキング CSV (tracking.csv, YOLO形式 正規化座標)
"""
import os
import sys
import yaml
import torch
import cv2
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.insert(0, os.path.dirname(__file__))

from wilt_tracker.models.model import DecoupledTracker
from wilt_tracker.utils.geometry import (
    calculate_iou,
    generate_heatmap,
    get_crop_224,
    map_local_to_global_vec
)
from wilt_tracker.utils.visualization import draw_target


class WiltTrackerEvaluator:
    """WiltTracker の検証モードパイプライン．"""

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Detection Model (YOLOv11)
        det_weights = self.cfg['detection']['weights']
        print(f"Loading Detection Model: {det_weights}")
        self.det_model = YOLO(det_weights)

        # Tracking Model (WiltTracker)
        trk_cfg = self.cfg['tracking']
        print(f"Loading Tracking Model: {trk_cfg['weights']}")
        self.track_model = DecoupledTracker(
            backbone=trk_cfg['backbone'],
            pretrained=False,
            past_frames=trk_cfg['past_frames'],
            use_conf=False,
            multi_head=trk_cfg['multi_head']
        ).to(self.device).eval()

        self.track_model.load_state_dict(
            torch.load(trk_cfg['weights'], map_location=self.device)
        )

    # ------------------------------------------------------------------
    # 画像フィルタリング
    # ------------------------------------------------------------------
    def _filter_by_time(self, img_dir, start_time, end_time):
        """時間帯でフィルタリングした画像パスリストを返す．"""
        all_paths = sorted(list(img_dir.glob("*.jpg")))
        filtered = []
        for p in all_paths:
            m = re.search(r'-(\d{4})\.jpg$', p.name)
            if m:
                t = m.group(1)
                if start_time <= t <= end_time:
                    filtered.append(p)
        return filtered

    # ------------------------------------------------------------------
    # 初期検出
    # ------------------------------------------------------------------
    def _detect_targets(self, img_bgr):
        """
        YOLOv11 で葉を検出し，WiltTracker でリファインした初期ターゲットを返す．
        切り出し領域 (crop_box) はここで固定され，以降変更しない．
        """
        det_cfg = self.cfg['detection']
        det_results = self.det_model(img_bgr, conf=det_cfg['conf_threshold'])[0]

        targets = []
        if not det_results.boxes:
            return targets

        all_boxes = det_results.boxes.xyxyn.cpu().numpy()

        # 面積上位 N 個を NMS で選択
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in all_boxes]
        sorted_idx = np.argsort(areas)[::-1]
        top_n = det_cfg.get('top_n', 5)

        picked = []
        for idx in sorted_idx:
            if len(picked) >= top_n:
                break
            if any(calculate_iou(all_boxes[idx], all_boxes[p]) > 0.05 for p in picked):
                continue
            picked.append(idx)

        # WiltTracker でリファイン
        for i, idx in enumerate(picked):
            box = all_boxes[idx]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            bw = box[2] - box[0]
            bh = box[3] - box[1]

            S = max(bw, bh) * 3.0
            crop_box = [cx, cy, S, S]

            crop_224 = get_crop_224(img_bgr, crop_box)
            tensor = torch.from_numpy(
                cv2.cvtColor(crop_224, cv2.COLOR_BGR2RGB)
            ).permute(2, 0, 1).unsqueeze(0).to(self.device, torch.uint8)

            hm_zeros = torch.zeros(1, 15, 224, 224, dtype=torch.uint8, device=self.device)
            x = torch.cat([tensor, hm_zeros], dim=1)

            with torch.no_grad():
                pred, _, _ = self.track_model(x, None)
                pred_local = pred[0].cpu().numpy().copy()

            vec = map_local_to_global_vec(pred_local, crop_box)
            targets.append({
                "id": i,
                "current_pos": vec.copy(),
                "history": [],
                "crop_box": crop_box,  # 固定
            })
            print(f"  ID {i} initialized")

        return targets

    # ------------------------------------------------------------------
    # 追跡
    # ------------------------------------------------------------------
    def _track_frame(self, img_bgr, targets, frame_idx):
        """全ターゲットを1フレーム分追跡する (crop_box 固定)．"""
        trk_cfg = self.cfg['tracking']
        stride = trk_cfg['past_stride']
        mixed = trk_cfg.get('use_mixed_stride', True)

        for target in targets:
            cb = target['crop_box']
            crop_224 = get_crop_224(img_bgr, cb)
            tensor = torch.from_numpy(
                cv2.cvtColor(crop_224, cv2.COLOR_BGR2RGB)
            ).permute(2, 0, 1).unsqueeze(0).to(self.device, torch.uint8)

            # ヒートマップ生成
            c_cx, c_cy, c_bw, c_bh = cb
            hms = []
            for p_num in range(1, 6):
                k = (1 if p_num == 1 else (p_num - 1) * stride) if mixed else p_num * stride
                p_idx = frame_idx - k
                if 0 <= p_idx < len(target['history']):
                    p = target['history'][p_idx]
                    pts = []
                    for xi, yi in [(0, 1), (4, 5), (6, 7)]:
                        lx = (p[xi] - (c_cx - c_bw / 2)) / c_bw
                        ly = (p[yi] - (c_cy - c_bh / 2)) / c_bh
                        pts.append((lx, ly))
                    for pt in pts:
                        hms.append(generate_heatmap(224, [pt]))
                else:
                    for _ in range(3):
                        hms.append(torch.zeros(224, 224, dtype=torch.uint8))

            hm_tensor = torch.stack(hms).unsqueeze(0).to(self.device)
            x = torch.cat([tensor, hm_tensor], dim=1).to(torch.uint8)

            with torch.no_grad():
                pred, _, _ = self.track_model(x, None)
                pred_local = pred[0].cpu().numpy().copy()

            vec = map_local_to_global_vec(pred_local, cb)
            if np.any(np.isnan(vec)):
                vec = target['current_pos'].copy()

            target['history'].append(vec.copy())
            target['current_pos'] = vec.copy()

    # ------------------------------------------------------------------
    # 出力: YOLO 形式 CSV
    # ------------------------------------------------------------------
    @staticmethod
    def _save_tracking_csv(csv_data, output_path):
        """
        YOLO 形式（正規化座標）の CSV を保存する．
        列: frame, leaf_id, cx, cy, w, h, base_x, base_y, tip_x, tip_y
        全て 0-1 正規化座標．
        """
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # 出力: 可視化動画
    # ------------------------------------------------------------------
    @staticmethod
    def _create_video(image_dir, output_path, fps=10):
        """可視化画像から MP4 動画を生成する．"""
        imgs = sorted(list(Path(image_dir).glob("*.jpg")))
        if not imgs:
            return
        first = cv2.imread(str(imgs[0]))
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        for img_path in imgs:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                writer.write(frame)
        writer.release()
        print(f"  Video saved: {output_path}")

    # ------------------------------------------------------------------
    # メインエントリポイント
    # ------------------------------------------------------------------
    def run(self):
        """各日を独立に処理する．"""
        inp_cfg = self.cfg['input']
        out_cfg = self.cfg['output']
        image_root = Path(inp_cfg['image_dir'])
        start_time = inp_cfg['time_range'][0].replace(":", "")
        end_time = inp_cfg['time_range'][1].replace(":", "")
        output_root = Path(out_cfg['dir'])

        save_images = out_cfg.get('save_images', True)
        save_video = out_cfg.get('save_video', True)
        save_csv = out_cfg.get('save_csv', True)

        # ディレクトリ判定
        direct_jpgs = list(image_root.glob("*.jpg"))
        if direct_jpgs:
            day_dirs = [image_root]
        else:
            day_dirs = sorted([
                d for d in image_root.iterdir()
                if d.is_dir() and not d.name.startswith('@')
            ])

        for day_dir in day_dirs:
            day_name = day_dir.name
            img_paths = self._filter_by_time(day_dir, start_time, end_time)
            if not img_paths:
                print(f"[{day_name}] No images. Skipping.")
                continue

            # 出力ディレクトリ名: {area}_{date} 形式を推定
            # ファイル名例: 33_04_HDR_20250507-0700.jpg → area=33, date=20250507
            first_name = img_paths[0].stem
            m = re.match(r'(\d+)_\d+_HDR_(\d+)-', first_name)
            if m:
                area_date = f"{m.group(1)}_{m.group(2)}"
            else:
                area_date = day_name

            day_out = output_root / area_date
            img_out = day_out / "images"
            img_out.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Processing: {area_date} ({len(img_paths)} frames)")
            print(f"{'='*60}")

            # 初期検出
            first_img = cv2.imread(str(img_paths[0]))
            if first_img is None:
                continue
            targets = self._detect_targets(first_img)
            if not targets:
                print(f"  No targets. Skipping.")
                continue
            print(f"  {len(targets)} targets detected. Tracking...")

            csv_rows = []

            for i, img_path in enumerate(tqdm(img_paths, desc=f"  {area_date}")):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue

                if i == 0:
                    for t in targets:
                        t['history'].append(t['current_pos'].copy())
                else:
                    self._track_frame(img_bgr, targets, i)

                # 可視化画像
                if save_images:
                    vis = img_bgr.copy()
                    for t in targets:
                        draw_target(vis, t['current_pos'], t['id'])
                    cv2.imwrite(str(img_out / f"{img_path.stem}_vis.jpg"), vis)

                # CSV 行 (YOLO形式: 正規化座標)
                if save_csv:
                    for t in targets:
                        p = t['current_pos']
                        csv_rows.append({
                            "frame": img_path.name,
                            "leaf_id": t['id'],
                            "cx": f"{p[0]:.6f}",
                            "cy": f"{p[1]:.6f}",
                            "w": f"{p[2]:.6f}",
                            "h": f"{p[3]:.6f}",
                            "base_x": f"{p[4]:.6f}",
                            "base_y": f"{p[5]:.6f}",
                            "tip_x": f"{p[6]:.6f}",
                            "tip_y": f"{p[7]:.6f}",
                        })

            # CSV 保存
            if save_csv and csv_rows:
                csv_path = day_out / "tracking.csv"
                self._save_tracking_csv(csv_rows, csv_path)
                print(f"  CSV saved: {csv_path}")

            # 動画生成
            if save_video and save_images:
                video_path = day_out / "video.mp4"
                self._create_video(img_out, video_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WiltTracker Evaluation Mode")
    parser.add_argument('--config', type=str, default='configs/eval.yaml')
    args = parser.parse_args()

    evaluator = WiltTrackerEvaluator(args.config)
    evaluator.run()
