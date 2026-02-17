"""
dataset_v2 作成スクリプト

legacy/annotations/ と legacy/images/ から WiltTracker 訓練用データセットを生成する.

入力:
  - legacy/annotations/{area}_{date}.csv
    列: frame_index, filename, leaf_id, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
        base_x, base_y, tip_x, tip_y, image_width, image_height, is_manual
    座標は全て 0-1 正規化.
  - legacy/images/{area}/{date}/{filename}

出力:
  - dataset_v2/images/{area}_{leafid}_{date}_{HHMM}.jpg  (切り出し画像, リサイズなし)
  - dataset_v2/labels/{area}_{leafid}_{date}_{HHMM}.txt  (YOLO形式ラベル)

ラベル形式:
  class cx cy w h base_x base_y tip_x tip_y
  全て切り出し領域に対する 0-1 正規化座標.

切り出し戦略:
  各葉 (area, date, leaf_id) について，7:00-17:00 の全フレームの
  BBox 中心と BBox サイズの平均を計算し，max(avg_w, avg_h) * 3.0 の
  正方形（正規化座標空間）を切り出し領域として固定する.
  この固定領域で全フレームを切り出す.
"""

import os
import glob
import csv
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm


def prepare_dataset(csv_dir, image_root, output_dir):
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"Loading CSVs from {csv_dir}")
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

    # (area, date, leaf_id) -> [records]
    data_map = defaultdict(list)

    for csv_path in csv_files:
        basename = os.path.basename(csv_path)
        try:
            area_id, date_str = basename.split('.')[0].split('_')
        except ValueError:
            continue

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['filename']
                try:
                    hhmm = img_name.split('-')[-1].replace('.jpg', '')
                    dt = datetime.strptime(f"{date_str}-{hhmm}", "%Y%m%d-%H%M")
                except (ValueError, IndexError):
                    continue

                item = {
                    'filename': img_name,
                    'timestamp': dt,
                    'bbox': [
                        float(row['bbox_xmin']), float(row['bbox_ymin']),
                        float(row['bbox_xmax']), float(row['bbox_ymax'])
                    ],
                    'base': [float(row['base_x']), float(row['base_y'])],
                    'tip': [float(row['tip_x']), float(row['tip_y'])],
                    'width': int(row['image_width']),
                    'height': int(row['image_height']),
                    'area_id': area_id,
                    'date': date_str,
                    'leaf_id': row['leaf_id']
                }
                data_map[(area_id, date_str, row['leaf_id'])].append(item)

    print(f"Found {len(data_map)} leaf tracks")

    total_images = 0
    for key, records in tqdm(data_map.items()):
        area_id, date_str, leaf_id = key

        # 7:00-17:00 のデータで統計を計算
        daily = [r for r in records if 7 <= r['timestamp'].hour <= 17]
        if not daily:
            continue

        # 全フレームの BBox 中心・サイズの平均
        centers_x = [(r['bbox'][0] + r['bbox'][2]) / 2 for r in daily]
        centers_y = [(r['bbox'][1] + r['bbox'][3]) / 2 for r in daily]
        widths = [r['bbox'][2] - r['bbox'][0] for r in daily]
        heights = [r['bbox'][3] - r['bbox'][1] for r in daily]

        avg_cx = np.mean(centers_x)
        avg_cy = np.mean(centers_y)
        avg_w = np.mean(widths)
        avg_h = np.mean(heights)

        # 固定切り出し領域: max(avg_w, avg_h) * 3.0 の正方形
        crop_size = max(avg_w, avg_h) * 3.0
        crop_x1 = avg_cx - crop_size / 2
        crop_y1 = avg_cy - crop_size / 2
        crop_x2 = avg_cx + crop_size / 2
        crop_y2 = avg_cy + crop_size / 2

        # 全フレーム（7:00前後含む）を切り出し
        for r in records:
            img_path = os.path.join(image_root, area_id, date_str, r['filename'])
            if not os.path.exists(img_path):
                continue

            try:
                img = Image.open(img_path).convert('RGB')
                W, H = img.size

                # 正規化座標 → ピクセル座標
                px1 = int(crop_x1 * W)
                py1 = int(crop_y1 * H)
                px2 = int(crop_x2 * W)
                py2 = int(crop_y2 * H)
                cw = px2 - px1
                ch = py2 - py1

                # 画像外にはみ出す場合は黒で埋める
                canvas = Image.new('RGB', (cw, ch), (0, 0, 0))
                ix1 = max(0, px1)
                iy1 = max(0, py1)
                ix2 = min(W, px2)
                iy2 = min(H, py2)

                if ix2 > ix1 and iy2 > iy1:
                    patch = img.crop((ix1, iy1, ix2, iy2))
                    canvas.paste(patch, (ix1 - px1, iy1 - py1))

                # 保存ファイル名
                ts_str = r['timestamp'].strftime("%H%M")
                base_name = f"{area_id}_{leaf_id}_{date_str}_{ts_str}"
                canvas.save(os.path.join(images_dir, f"{base_name}.jpg"))

                # ラベル生成（切り出し領域に対する相対正規化座標）
                gx1, gy1, gx2, gy2 = r['bbox']
                rcx = ((gx1 + gx2) / 2 - crop_x1) / crop_size
                rcy = ((gy1 + gy2) / 2 - crop_y1) / crop_size
                rw = (gx2 - gx1) / crop_size
                rh = (gy2 - gy1) / crop_size

                rbx = (r['base'][0] - crop_x1) / crop_size
                rby = (r['base'][1] - crop_y1) / crop_size
                rtx = (r['tip'][0] - crop_x1) / crop_size
                rty = (r['tip'][1] - crop_y1) / crop_size

                with open(os.path.join(labels_dir, f"{base_name}.txt"), 'w') as lf:
                    lf.write(f"0 {rcx:.6f} {rcy:.6f} {rw:.6f} {rh:.6f} "
                             f"{rbx:.6f} {rby:.6f} {rtx:.6f} {rty:.6f}\n")

                total_images += 1

            except Exception as e:
                print(f"Error: {img_path}: {e}")
                continue

    print(f"Dataset created: {total_images} samples in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset_v2 作成')
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='legacy/annotations のパス')
    parser.add_argument('--image_root', type=str, required=True,
                        help='legacy/images のパス')
    parser.add_argument('--output_dir', type=str, default='dataset_v2',
                        help='出力ディレクトリ')
    args = parser.parse_args()

    prepare_dataset(args.csv_dir, args.image_root, args.output_dir)
