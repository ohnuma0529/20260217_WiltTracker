import torch
import torch.nn as nn
from torchvision.ops import complete_box_iou_loss
import numpy as np


class WingLoss(nn.Module):
    def __init__(self, w=0.045, epsilon=0.01):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * np.log(1 + self.w / self.epsilon)

    def forward(self, x, y):
        diff = torch.abs(x - y)
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        return torch.mean(loss)


def generate_gaussian_target(target_reg, size=7, sigma=1.0):
    """
    target_reg: [B, 8] (cx, cy, w, h, bx, by, tx, ty)
    returns: [B, 3, 7, 7]
    """
    B = target_reg.shape[0]
    device = target_reg.device
    
    # Extract [B, 3, 2] points: (cx, cy), (bx, by), (tx, ty)
    p1 = target_reg[:, 0:2]
    p2 = target_reg[:, 4:6]
    p3 = target_reg[:, 6:8]
    points = torch.stack([p1, p2, p3], dim=1) # [B, 3, 2]

    yy, xx = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
    yy = yy.view(1, 1, size, size).float()
    xx = xx.view(1, 1, size, size).float()
    
    # Scale to grid size [0, size-1]
    grid_points = points * (size - 1)
    px = grid_points[:, :, 0].view(B, 3, 1, 1)
    py = grid_points[:, :, 1].view(B, 3, 1, 1)
    
    dist_sq = (xx - px)**2 + (yy - py)**2
    gaussian = torch.exp(-dist_sq / (2 * sigma**2))
    return gaussian


class ImprovedLoss(nn.Module):
    def __init__(self, w_box=5.0, w_kpt=20.0, w_conf=1.0, w_temporal=1.0, w_attn=10.0, use_wing=True):
        super().__init__()
        self.w_box = w_box
        self.w_kpt = w_kpt
        self.w_conf = w_conf
        self.w_temporal = w_temporal
        self.w_attn = w_attn
        self.w_consistency = 0.0 # Default
        
        self.l1 = nn.L1Loss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='none') 
        self.mse_mean = nn.MSELoss(reduction='mean')
        self.wing = WingLoss() if use_wing else None

    def forward(self, pred_reg, pred_conf, target_reg, masks=None, pred_reg_clean=None, temporal_target=None, prev_conf_mask=None):
        """
        pred_reg: [B, 8] (cx, cy, w, h, bx, by, tx, ty)
        pred_conf: [B, 1] (logits)
        target_reg: [B, 8]
        masks: [B, 3, 7, 7] (Attention masks)
        pred_reg_clean: [B, 8]
        temporal_target: [B, 8]
        prev_conf_mask: [B, 1]
        """
        # 1. BBox Loss (CIoU)
        p_box = pred_reg[:, :4]
        t_box = target_reg[:, :4]
        
        def xywh2xyxy(x):
            cx, cy, w, h = x.unbind(-1)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            return torch.stack([x1, y1, x2, y2], dim=-1)
        
        p_xyxy = xywh2xyxy(p_box)
        t_xyxy = xywh2xyxy(t_box)
        
        loss_box = complete_box_iou_loss(p_xyxy, t_xyxy, reduction='mean')
        
        # 2. Keypoint Loss (Wing or L1)
        p_kpt = pred_reg[:, 4:]
        t_kpt = target_reg[:, 4:]
        if self.wing:
            loss_kpt = self.wing(p_kpt, t_kpt)
        else:
            loss_kpt = self.l1(p_kpt, t_kpt)
        
        # 3. Confidence Loss (BCE)
        loss_conf = torch.tensor(0.0).to(pred_reg.device)
        if pred_conf is not None:
            target_conf = torch.full_like(pred_conf, 0.9)
            loss_conf = self.bce(pred_conf, target_conf)

        # 4. Temporal Smoothness Loss
        loss_temporal = torch.tensor(0.0).to(pred_reg.device)
        if temporal_target is not None and prev_conf_mask is not None:
            sq_diff = self.mse(pred_reg, temporal_target) 
            loss_temporal = (sq_diff * prev_conf_mask).mean()

        # 5. Consistency Loss (Optional)
        loss_consistency = torch.tensor(0.0).to(pred_reg.device)
        if pred_reg_clean is not None:
            loss_consistency = self.l1(pred_reg, pred_reg_clean)

        # 6. Attention Supervision Loss (Optional)
        loss_attn = torch.tensor(0.0).to(pred_reg.device)
        if masks is not None and self.w_attn > 0:
            target_masks = generate_gaussian_target(target_reg, size=7, sigma=1.0)
            if masks.shape[1] == 1:
                # If model only has one mask, collapse target into one (average of 3 pts)
                target_masks = target_masks.mean(dim=1, keepdim=True)
            loss_attn = self.mse_mean(masks, target_masks)

        # Total
        total_loss = (self.w_box * loss_box) + \
                     (self.w_kpt * loss_kpt) + \
                     (self.w_conf * loss_conf) + \
                     (self.w_temporal * loss_temporal) + \
                     (self.w_consistency * loss_consistency) + \
                     (self.w_attn * loss_attn)
        
        return total_loss, {
            'loss_box': loss_box.item(),
            'loss_kpt': loss_kpt.item(),
            'loss_conf': loss_conf.item(),
            'loss_temp': loss_temporal.item(),
            'loss_const': loss_consistency.item(),
            'loss_attn': loss_attn.item()
        }
