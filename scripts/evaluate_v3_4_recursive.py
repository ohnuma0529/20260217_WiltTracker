import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import os
import sys
import glob
from tqdm import tqdm
import argparse
import pandas as pd

# Add v3.4_dev to path to find local wilt_tracker
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wilt_tracker.models.model import DecoupledTracker

def generate_heatmap_recursive(size, points, sigma=3):
    hm = np.zeros((size, size), dtype=np.uint8)
    for pt in points:
        x, y = pt
        if x < 0 or x >= 1 or y < 0 or y >= 1: continue
        px, py = int(x * size), int(y * size)
        radius = sigma * 3
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                ix, iy = px + i, py + j
                if 0 <= ix < size and 0 <= iy < size:
                    dist_sq = i**2 + j**2
                    val = np.exp(-dist_sq / (2 * sigma**2))
                    hm[iy, ix] = max(hm[iy, ix], int(val * 255))
    return torch.from_numpy(hm)

def draw_styled_prediction(img, vec, color, label, is_gt=True):
    h, w = img.shape[:2]
    cx, cy, nw, nh = vec[0:4]
    x1, y1 = int((cx-nw/2)*w), int((cy-nh/2)*h)
    x2, y2 = int((cx+nw/2)*w), int((cy+nh/2)*h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1 if is_gt else 2)
    bx, by = int(vec[4]*w), int(vec[5]*h)
    cv2.circle(img, (bx, by), 4, color, -1 if is_gt else 1)
    tx, ty = int(vec[6]*w), int(vec[7]*h)
    cv2.circle(img, (tx, ty), 6, color, -1 if is_gt else 2)
    cv2.line(img, (bx, by), (tx, ty), color, 1)

def get_pixel_err(p, t):
    return np.linalg.norm(p - t) * 224

def visualize_leaf_sequence(model, dataset_dir, area_id, leaf_id, date, output_subdir, device, stride=5, use_mixed_stride=True, output_base='results/recursive_eval', save_freq=1):
    model.eval()
    pattern = os.path.join(dataset_dir, 'labels', f"{area_id}_{leaf_id}_{date}_*.txt")
    label_files = sorted(glob.glob(pattern))
    if not label_files: return None
    
    gts = []
    for lf in label_files:
        with open(lf, 'r') as f:
            line = f.readline().strip().split()
            if not line: continue
            gts.append(np.array([float(x) for x in line[1:9]]))
            
    past_preds = []
    metrics = []
    output_dir = os.path.join(output_base, output_subdir, f"{area_id}_{leaf_id}_{date}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Visualizing leaf {leaf_id} to {output_dir}... (Save Freq: {save_freq})")
    
    with torch.no_grad():
        for i in tqdm(range(len(gts))):
            base_name = os.path.basename(label_files[i]).replace('.txt', '')
            img_path = os.path.join(dataset_dir, 'images', f"{base_name}.jpg")
            if not os.path.exists(img_path): continue
            
            img_pil = Image.open(img_path).convert('RGB')
            img_panel = cv2.cvtColor(np.array(img_pil.resize((224, 224))), cv2.COLOR_RGB2BGR)
            
            # Predict
            img_tensor = torch.from_numpy(np.array(img_pil.resize((224, 224)))).permute(2, 0, 1).to(device).unsqueeze(0).to(torch.uint8)
            hms = []
            for p_num in range(1, 6):
                k = (1 if p_num == 1 else (p_num-1)*stride) if use_mixed_stride else p_num*stride
                p_idx = i - k
                if p_idx >= 0:
                    p = past_preds[p_idx]
                    pts = [(p[0], p[1]), (p[4], p[5]), (p[6], p[7])]
                    for pt in pts: hms.append(generate_heatmap_recursive(224, [pt]))
                else:
                    for _ in range(3): hms.append(torch.zeros(224, 224))
            
            hm_tensor = torch.stack(hms).to(device).unsqueeze(0).to(torch.uint8)
            x_input = torch.cat([img_tensor, hm_tensor], dim=1)
            
            # model returns (reg, conf, masks)
            pred_reg, _, _ = model(x_input, None)
            pred_np = pred_reg[0].cpu().numpy()
            target = gts[i]
            past_preds.append(pred_np)
            
            # Metrics
            b_err = get_pixel_err(pred_np[4:6], target[4:6])
            t_err = get_pixel_err(pred_np[6:8], target[6:8])
            
            metrics.append({
                'frame': i,
                'base_err': b_err,
                'tip_err': t_err,
                'mean_err': (b_err + t_err) / 2
            })
            
            # Draw
            if i % save_freq == 0:
                draw_styled_prediction(img_panel, target, (0, 255, 0), "GT", is_gt=True)
                draw_styled_prediction(img_panel, pred_np, (0, 0, 255), "Pred", is_gt=False)
                cv2.imwrite(os.path.join(output_dir, f"{i:04d}.jpg"), img_panel)
                
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    
    summary = {
        'avg_base_err': df['base_err'].mean(),
        'avg_tip_err': df['tip_err'].mean(),
        'final_base_err': df['base_err'].iloc[-1],
        'final_tip_err': df['tip_err'].iloc[-1],
        'pck_15': (df['mean_err'] < 15).mean() * 100
    }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='densenet121')
    parser.add_argument('--multi_head', action='store_true', default=False)
    parser.add_argument('--no_multi_head', action='store_false', dest='multi_head')
    parser.add_argument('--no_conf', action='store_true', default=False)
    parser.add_argument('--output_subdir', type=str, default='eval_run')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--mixed_stride', action='store_true', default=True)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--leaf_ids', type=str, default='0')
    parser.add_argument('--dataset_dir', type=str, default='/home/ohnuma/20251228_WiltTracker/dataset_v2')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DecoupledTracker(
        backbone=args.backbone,
        pretrained=False,
        past_frames=5,
        use_conf=not args.no_conf,
        multi_head=args.multi_head
    ).to(device)
    
    print(f"Loading weights from {args.weights}...")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    
    leaf_ids = [int(x) for x in args.leaf_ids.split(',')]
    all_summaries = []
    
    # Area 34 and Area 32 as per train_v3.4.py logic
    for area in ['34', '32']:
        for lid in leaf_ids:
            res = visualize_leaf_sequence(
                model, args.dataset_dir, area, lid, '20251114' if area == '34' else '20251112',
                args.output_subdir, device, 
                stride=args.stride, use_mixed_stride=args.mixed_stride,
                save_freq=args.save_freq
            )
            if res:
                res['area'] = area
                res['leaf_id'] = lid
                all_summaries.append(res)
                
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        print("\n--- Recursive Evaluation Summary ---")
        print(summary_df.to_string(index=False))
        # Save global summary in subdir
        summary_df.mean(numeric_only=True).to_frame().T.to_csv(
            os.path.join('results/recursive_eval', args.output_subdir, "summary.csv"), index=False
        )

if __name__ == '__main__':
    main()
