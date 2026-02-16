from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import cv2

def plot_results(file='results.csv', dir=''):
    """
    Plots training results from a CSV file.
    Expects CSV with columns like 'epoch', 'train_loss', 'val_loss', etc.
    """
    save_dir = Path(file).parent if not dir else Path(dir)
    try:
        data = pd.read_csv(file)
        x = data['epoch']
        
        components = [
            ('Total Loss', 'train_loss', 'val_loss'),
            ('Box Loss', 'box_loss', 'val_box_loss'),
            ('Keypoint Loss', 'kpt_loss', 'val_kpt_loss'),
            ('Consistency Loss', 'const_loss', 'val_const_loss'),
            ('Attention Loss', 'loss_attn', None),
            ('Recursive Error', 'reward_err', None)
        ]
        
        # Filter existing columns
        plot_list = []
        for title, t_col, v_col in components:
            if t_col in data.columns:
                plot_list.append((title, t_col, v_col if v_col in data.columns else None))
        
        num_plots = len(plot_list)
        rows = (num_plots + 2) // 3
        cols = min(num_plots, 3)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = np.array(axes).flatten()
        
        for i, (title, t_col, v_col) in enumerate(plot_list):
            ax = axes[i]
            ax.plot(x, data[t_col], label=f'Train {title}')
            if v_col:
                ax.plot(x, data[v_col], label=f'Val {title}')
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        fig.tight_layout()
        fig.savefig(save_dir / 'results.png', dpi=200)
        plt.close()
        
        # Add LR plot if available (Optional, maybe on twinx of ax[0]?)
        # Let's keep it simple as requested: Box, Kpt, Conf on Right.
        
        # You can add valid IoU or other metrics if logged
        # For now just loss
        
        fig.tight_layout()
        fig.savefig(save_dir / 'results.png', dpi=200)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

from pathlib import Path

def plot_batch_input(images, targets, paths=None, fname='images.jpg'):
    """
    Plots a grid of the batch images with targets overlaid.
    images: [B, 3, H, W] - Normalized tensor
    targets: [B, 8] - Normalized coords
    """
    # Slice first on GPU to save CPU memory transform
    if images.shape[0] > 16:
        images = images[:16]
        targets = targets[:16]

    # Handle uint8 vs float32
    if images.dtype == torch.uint8:
        # Already 0-255, just pick first 3 channels (RGB)
        imgs = images[:, :3, :, :].float() / 255.0
    else:
        # Denormalize (Assuming float32 normalized 0-1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        imgs = images[:, :3, :, :] * std + mean
    
    # To numpy
    imgs = imgs.cpu().numpy() # [B, 3, H, W]
    targets = targets.cpu().numpy()
    
    bs, _, h, w = imgs.shape
    rows = int(np.sqrt(bs))
    cols = int(np.ceil(bs / rows))

    mosaic = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)

    for i in range(bs):
        img = imgs[i].transpose(1, 2, 0) # [H, W, 3]
        img = np.ascontiguousarray(img * 255, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 1. Overlay Heatmaps (Optional diagnosis)
        # images may contain 18 channels (3 RGB + 15 HM)
        if images.shape[1] > 3:
            # Extract first past frame heatmaps (channels 3,4,5)
            # Center (3), Base (4), Tip (5)
            hm_past = images[i, 3:6, :, :].cpu().numpy() # [3, H, W]
            overlay = np.zeros_like(img)
            # Magenta for Center (R+B)
            overlay[:, :, 0] = np.maximum(overlay[:, :, 0], hm_past[0]) # B
            overlay[:, :, 2] = np.maximum(overlay[:, :, 2], hm_past[0]) # R
            # Cyan for Keypoints (G+B)
            overlay[:, :, 0] = np.maximum(overlay[:, :, 0], hm_past[1]) # B
            overlay[:, :, 1] = np.maximum(overlay[:, :, 1], hm_past[1]) # G
            overlay[:, :, 0] = np.maximum(overlay[:, :, 0], hm_past[2]) # B
            overlay[:, :, 1] = np.maximum(overlay[:, :, 1], hm_past[2]) # G
            
            img = cv2.addWeighted(img, 1.0, overlay, 0.5, 0)

        # 2. Draw GT
        t = targets[i] # cx, cy, w, h, bx, by, tx, ty (all 0-1)
        
        # BBox (Red)
        x1 = int((t[0] - t[2]/2) * w)
        y1 = int((t[1] - t[3]/2) * h)
        x2 = int((t[0] + t[2]/2) * w)
        y2 = int((t[1] + t[3]/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # BGR: Red
        
        # Keypoints
        bx, by = int(t[4] * w), int(t[5] * h)
        tx, ty = int(t[6] * w), int(t[7] * h)
        
        # Base (Red Point)
        if bx > 0 and by > 0:
            cv2.circle(img, (bx, by), 4, (0, 0, 255), -1) # BGR: Red
        
        # Tip (Blue Point)
        if tx > 0 and ty > 0:
            cv2.circle(img, (tx, ty), 4, (255, 0, 0), -1) # BGR: Blue
        
        # Line (Green)
        if bx > 0 and by > 0 and tx > 0 and ty > 0:
            cv2.line(img, (bx, by), (tx, ty), (0, 255, 0), 2) # BGR: Green
        
        r = i // cols
        c = i % cols
        mosaic[r*h:(r+1)*h, c*w:(c+1)*w, :] = img
        
    cv2.imwrite(str(fname), mosaic)


def plot_batch_pred(images, targets, preds, fname='pred_batch.jpg'):
    """
    Plots a grid of the batch images with targets (Green) and predictions (Red) overlaid.
    """
    # Slice on GPU
    if images.shape[0] > 16:
        images = images[:16]
        targets = targets[:16]
        preds = preds[:16]

    # Handle uint8 vs float32
    if images.dtype == torch.uint8:
        imgs = images[:, :3, :, :].float() / 255.0
    else:
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        imgs = images[:, :3, :, :] * std + mean
    
    # To numpy
    imgs = imgs.cpu().numpy() # [B, 3, H, W]
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()

    bs, _, h, w = imgs.shape
    rows = int(np.sqrt(bs))
    cols = int(np.ceil(bs / rows))

    mosaic = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)

    for i in range(bs):
        img = imgs[i].transpose(1, 2, 0) # [H, W, 3]
        img = np.ascontiguousarray(img * 255, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Draw GT (Green)
        t = targets[i]
        x1 = int((t[0] - t[2]/2) * w)
        y1 = int((t[1] - t[3]/2) * h)
        x2 = int((t[0] + t[2]/2) * w)
        y2 = int((t[1] + t[3]/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw Pred (Red)
        p = preds[i] # cx, cy, w, h, bx, by, tx, ty (all 0-1)
        if np.isnan(p).any():
            continue
        px1 = int((p[0] - p[2]/2) * w)
        py1 = int((p[1] - p[3]/2) * h)
        px2 = int((p[0] + p[2]/2) * w)
        py2 = int((p[1] + p[3]/2) * h)
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
        
        # Keypoints Pred
        if len(p) >= 8:
            cv2.circle(img, (int(p[4]*w), int(p[5]*h)), 3, (0, 0, 255), -1) # Base
            cv2.circle(img, (int(p[6]*w), int(p[7]*h)), 3, (0, 0, 255), -1) # Tip

        r = i // cols
        c = i % cols
        mosaic[r*h:(r+1)*h, c*w:(c+1)*w, :] = img
        
    cv2.imwrite(str(fname), mosaic)
