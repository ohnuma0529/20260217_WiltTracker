import torch
import numpy as np
import cv2

def calculate_iou(box1, box2):
    # box: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union == 0: return 0
    return intersection / union

def generate_heatmap(size, points, sigma=3):
    """
    点リスト(0-1正規化)からヒートマップを生成する．
    """
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

def get_crop_224(img_bgr, crop_box):
    """
    指定された座標 [cx, cy, cw, ch] (正規化) で画像をクロップし，
    224x224にリサイズして返す．画像外領域はゼロパディング（黒）する．
    """
    h, w = img_bgr.shape[:2]
    cx, cy, cw, ch = crop_box
    x1 = int((cx - cw/2) * w)
    y1 = int((cy - ch/2) * h)
    x2 = int((cx + cw/2) * w)
    y2 = int((cy + ch/2) * h)

    pad_x1 = max(0, -x1)
    pad_y1 = max(0, -y1)
    pad_x2 = max(0, x2 - w)
    pad_y2 = max(0, y2 - h)

    crop = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
        crop = cv2.copyMakeBorder(crop, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    return cv2.resize(crop, (224, 224))

def map_local_to_global_vec(pred_local, crop_box):
    """
    クロップ内相対座標 [0, 1] を，元の画像全体の [0, 1] 座標系に変換する．
    pred_local: [cx, cy, w, h, bx, by, tx, ty]
    crop_box: [c_cx, c_cy, c_cw, c_ch]
    """
    c_cx, c_cy, c_cw, c_ch = crop_box
    pred_global = pred_local.copy()
    
    crop_left = c_cx - c_cw/2
    crop_top = c_cy - c_ch/2
    
    # 座標点 (cx, bx, tx)
    for i in [0, 4, 6]:
        pred_global[i] = pred_local[i] * c_cw + crop_left
    # 座標点 (cy, by, ty)
    for i in [1, 5, 7]:
        pred_global[i] = pred_local[i] * c_ch + crop_top
    # サイズ (w, h) - オフセットなし，比率のみ
    pred_global[2] = pred_local[2] * c_cw
    pred_global[3] = pred_local[3] * c_ch
    
    return pred_global
