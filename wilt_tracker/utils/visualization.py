import cv2
import numpy as np

def draw_target(img, vec, lid):
    """
    オリジナル解像度の画像上に追跡ターゲットを描画する．
    vec: [cx, cy, w, h, bx, by, tx, ty] (正規化座標)
    """
    h, w = img.shape[:2]
    cx, cy, nw, nh = vec[0:4]
    x1, y1 = int((cx-nw/2)*w), int((cy-nh/2)*h)
    x2, y2 = int((cx+nw/2)*w), int((cy+nh/2)*h)
    
    # Bbox (Red)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, f"ID:{lid}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Keypoints
    bx, by = int(vec[4]*w), int(vec[5]*h)
    tx, ty = int(vec[6]*w), int(vec[7]*h)
    cv2.circle(img, (bx, by), 5, (0, 0, 255), -1) # Base: Red
    cv2.circle(img, (tx, ty), 5, (255, 0, 0), -1) # Tip: Blue
    cv2.line(img, (bx, by), (tx, ty), (0, 255, 0), 2)

def draw_target_224(img, vec, lid):
    """
    224x224のデバッグ画像上にターゲットを描画する（線などの太さを調整）．
    """
    h, w = 224, 224
    cx, cy, nw, nh = vec[0:4]
    x1, y1 = int((cx-nw/2)*w), int((cy-nh/2)*h)
    x2, y2 = int((cx+nw/2)*w), int((cy+nh/2)*h)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.putText(img, f"ID:{lid}", (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    bx, by = int(vec[4]*w), int(vec[5]*h)
    tx, ty = int(vec[6]*w), int(vec[7]*h)
    cv2.circle(img, (bx, by), 3, (0, 0, 255), -1)
    cv2.circle(img, (tx, ty), 3, (255, 0, 0), -1)
    cv2.line(img, (bx, by), (tx, ty), (0, 255, 0), 1)

def apply_heatmap_overlay(img_bgr, hm_tensor, alpha=0.3):
    """
    ヒートマップテンソル(N, H, W)を合算し，JETカラーマップでオーバーレイする．
    """
    # hm_tensor shape: (15, 224, 224) または (1, 15, 224, 224)
    if hm_tensor.dim() == 4:
        hm = hm_tensor[0]
    else:
        hm = hm_tensor
        
    combined_hm = hm.sum(dim=0).cpu().numpy()
    combined_hm = (combined_hm / (combined_hm.max() + 1e-6) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(combined_hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
