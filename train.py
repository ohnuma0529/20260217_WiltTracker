
import os
import argparse
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Add local v3.4 copy of wilt_tracker to path
sys.path.insert(0, os.path.dirname(__file__))

from wilt_tracker.data.dataset import LeafTrackingDatasetV2
from wilt_tracker.models.model import DecoupledTracker
from wilt_tracker.models.loss import ImprovedLoss
from wilt_tracker.utils.general import increment_path
from wilt_tracker.utils.plots import plot_batch_pred, plot_batch_input, plot_results

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class StageEarlyStopping:
    """Early stopping specific to a curriculum stage."""
    def __init__(self, patience=10, min_delta=0.0, min_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.epoch_counter = 0
        self.patience_counter = 0
        self.best_loss = float('inf')
        self.converged = False

    def __call__(self, val_loss):
        self.epoch_counter += 1
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.epoch_counter >= self.min_epochs and self.patience_counter >= self.patience:
            self.converged = True
        return self.converged

class RewardEarlyStopping:
    """Early stopping based on recursive tracking rewards (errors)."""
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_reward = float('inf') # Reward here is "error", so lower is better
        self.early_stop = False

    def __call__(self, reward):
        if reward < self.best_reward - self.min_delta:
            self.best_reward = reward
            self.counter = 0
            return True # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False # Not improved

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

def save_attention_snapshot(model, device, save_dir, epoch, dataset_dir, img_size=224):
    """Saves a quick snapshot of attention masks for diagnostic."""
    model.eval()
    # Use Area 34, Leaf 0, 08:00 as representative
    area, leaf, date, target_time = '34', 0, '20251114', '0800'
    img_path = os.path.join(dataset_dir, 'images', f"{area}_{leaf}_{date}_{target_time}.jpg")
    if not os.path.exists(img_path): return
    
    img_pil = Image.open(img_path).convert('RGB')
    img_np = np.array(img_pil.resize((img_size, img_size)))
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device).to(torch.uint8)
    
    # Real past GT history for snapshot
    from datetime import datetime, timedelta
    hms = []
    curr_t = target_time
    for _ in range(5):
        # 5-min stride
        dt = datetime.strptime(curr_t, "%H%M") - timedelta(minutes=5)
        curr_t = dt.strftime("%H%M")
        lf = os.path.join(dataset_dir, 'labels', f"{area}_{leaf}_{date}_{curr_t}.txt")
        if os.path.exists(lf):
            with open(lf, 'r') as f:
                line = f.readline().strip().split()
                if line:
                    p = [float(x) for x in line[1:9]]
                    pts = [(p[0], p[1]), (p[4], p[5]), (p[6], p[7])]
                    for pt in pts: hms.append(generate_heatmap_recursive(img_size, [pt]))
                else:
                    for _ in range(3): hms.append(torch.zeros(img_size, img_size))
        else:
            for _ in range(3): hms.append(torch.zeros(img_size, img_size))
            
    hm_input = torch.stack(hms).to(device).unsqueeze(0).to(torch.uint8)
    x_input = torch.cat([img_tensor, hm_input], dim=1)
    
    with torch.no_grad():
        _, _, masks = model(x_input, None)
    
    if masks is not None:
        masks_np = masks[0].cpu().numpy()
        if len(masks_np.shape) == 2:
            # Single channel mask (e.g. single-head model)
            masks_np = masks_np[np.newaxis, ...]
        
        snap_dir = save_dir / 'evolution'
        snap_dir.mkdir(parents=True, exist_ok=True)
        names = ['bbox', 'base', 'tip']
        for i, name in enumerate(names):
            if i >= masks_np.shape[0]: break
            m = masks_np[i]
            m_res = cv2.resize(m, (img_size, img_size))
            m_min, m_max = m_res.min(), m_res.max()
            if m_max > m_min:
                m_res = (m_res - m_min) / (m_max - m_min)
            m_norm = (m_res * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(m_norm, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
            cv2.imwrite(str(snap_dir / f'epoch_{epoch}_{name}.jpg'), overlay)

def perform_recursive_validation(model, device, config, dataset_dir=None, area_id='34', leaf_id=0, date='20251114'):
    if dataset_dir is None:
        dataset_dir = config.get('data', {}).get('dataset_dir', 'dataset_v2')
    """Fast recursive evaluation on a single leaf to get reward (error)."""
    model.eval()
    import glob
    pattern = os.path.join(dataset_dir, 'labels', f"{area_id}_{leaf_id}_{date}_*.txt")
    label_files = sorted(glob.glob(pattern))
    if not label_files: return float('inf'), 0, 0
    
    gts = []
    for lf in label_files:
        with open(lf, 'r') as f:
            line = f.readline().strip().split()
            if not line: continue
            gts.append(np.array([float(x) for x in line[1:9]]))
            
    past_preds = []
    total_tip_err = 0
    total_base_err = 0
    stride = config['data'].get('past_stride', 5)
    use_mixed_stride = config['data'].get('use_mixed_stride', True)
    
    with torch.no_grad():
        for i in range(len(gts)):
            base_name = os.path.basename(label_files[i]).replace('.txt', '')
            img_path = os.path.join(dataset_dir, 'images', f"{base_name}.jpg")
            if not os.path.exists(img_path): continue
            
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = torch.from_numpy(np.array(img_pil.resize((224, 224)))).permute(2, 0, 1).to(device).unsqueeze(0).to(torch.uint8)
            
            hms = []
            for p_num in range(1, 6):
                k = (1 if p_num == 1 else (p_num-1)*stride) if use_mixed_stride else p_num*stride
                p_idx = i - k
                if p_idx >= 0:
                    p = past_preds[p_idx]
                    # [cx, cy, w, h, bx, by, tx, ty]
                    pts = [(p[0], p[1]), (p[4], p[5]), (p[6], p[7])]
                    for pt in pts: hms.append(generate_heatmap_recursive(224, [pt]))
                else:
                    for _ in range(3): hms.append(torch.zeros(224, 224))
            
            hm_tensor = torch.stack(hms).to(device).unsqueeze(0).to(torch.uint8)
            x_input = torch.cat([img_tensor, hm_tensor], dim=1)
            
            pred_reg, _, _ = model(x_input, None)
            
            pred_np = pred_reg[0].cpu().numpy()
            target = gts[i]
            past_preds.append(pred_np)
            
            # Accumulated error (pixel)
            total_base_err += np.linalg.norm(pred_np[4:6] - target[4:6]) * 224
            total_tip_err += np.linalg.norm(pred_np[6:8] - target[6:8]) * 224
            
    avg_base = total_base_err / len(gts)
    avg_tip = total_tip_err / len(gts)
    
    # Configurable Reward for v3.4 Ablation (Default 1.0 : 1.0)
    w_tip = config['evaluation'].get('reward_tip_weight', 1.0)
    w_base = config['evaluation'].get('reward_base_weight', 1.0)
    reward = (w_base * avg_base + w_tip * avg_tip) / (w_base + w_tip)
    return reward, avg_base, avg_tip

def run_comparison_eval(model, val_loader, device, save_dir):
    """
    Performs post-training evaluation to compare With-Past vs No-Past.
    Saves results to results_comparison.csv
    """
    print("\nRunning Post-Training Comparison Evaluation...")
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating Context Influence"):
            img = batch['image'].to(device)
            hm = batch['heatmap'].to(device)
            target = batch['target'].to(device)
            B = img.shape[0]
            
            # 1. With Past
            prev_conf = torch.ones(B, 1).to(device)
            x_past = torch.cat([img, hm], dim=1)
            pred_past, _, _ = model(x_past, prev_conf)
            
            # 2. No Past (Zeros)
            x_nopast = torch.cat([img, torch.zeros_like(hm)], dim=1)
            pred_nopast, _, _ = model(x_nopast, prev_conf)
            
            # 3. Calculate Errors (Pixel distance assuming 224x224)
            # pred: [B, 8] -> [cx, cy, w, h, bx, by, tx, ty]
            def get_pixel_err(p, t):
                # Only keypoints: 4:8
                p_pts = p[:, 4:].cpu().numpy() * 224.0
                t_pts = t[:, 4:].cpu().numpy() * 224.0
                # Base distance
                base_dist = np.sqrt(np.sum((p_pts[:, 0:2] - t_pts[:, 0:2])**2, axis=1))
                # Tip distance
                tip_dist = np.sqrt(np.sum((p_pts[:, 2:4] - t_pts[:, 2:4])**2, axis=1))
                return base_dist, tip_dist

            err_p_base, err_p_tip = get_pixel_err(pred_past, target)
            err_n_base, err_n_tip = get_pixel_err(pred_nopast, target)
            
            for i in range(B):
                results.append({
                    'base_err_past': err_p_base[i],
                    'tip_err_past': err_p_tip[i],
                    'base_err_nopast': err_n_base[i],
                    'tip_err_nopast': err_n_tip[i]
                })
    
    df = pd.DataFrame(results)
    summary = df.mean().to_frame().T
    summary['avg_err_past'] = (summary['base_err_past'] + summary['tip_err_past']) / 2
    summary['avg_err_nopast'] = (summary['base_err_nopast'] + summary['tip_err_nopast']) / 2
    summary['improvement_%'] = (summary['avg_err_nopast'] - summary['avg_err_past']) / summary['avg_err_nopast'] * 100
    
    comp_file = save_dir / 'results_comparison.csv'
    summary.to_csv(comp_file, index=False)
    print(f"\n--- Post-Training Comparison Summary ---")
    print(summary.to_string(index=False))
    print(f"Results saved to {comp_file}")


def train_v2(args):
    config_path = args.config
    # Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Global Seed
    seed = args.seed if args.seed is not None else config['experiment'].get('seed', 42)
    set_seed(seed)
    print(f"Set global seed to: {seed}")
    
    # Override from CLI if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    device = torch.device(config['training']['device'])
    print(f"Using device: {device}")
    
    # Setup Dirs
    save_dir = Path(increment_path(Path(config['experiment']['project']) / config['experiment']['name'], exist_ok=args.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
        
    # 2. Dataset / Dataloader
    print("Loading V2 datasets...")
    # Datasets
    train_cfg = config # Use config directly as train_cfg
    past_frames = train_cfg['model'].get('past_frames', 5)
    past_jitter_prob = train_cfg['data'].get('past_jitter_prob', 1.0)
    allow_val_jitter = train_cfg['data'].get('allow_val_jitter', False)
    use_mixed_stride = train_cfg['data'].get('use_mixed_stride', False)
    w_consistency_global = train_cfg['training'].get('w_consistency', 0.0)
    
    # Consistency loss requires clean heatmap
    use_clean_hm = w_consistency_global > 0
    
    train_ds = LeafTrackingDatasetV2(
        dataset_dir=train_cfg['data']['dataset_dir'],
        mode='train',
        img_size=tuple(train_cfg['data']['img_size']),
        past_window_minutes=train_cfg['data'].get('past_window_minutes', 30),
        past_frames=train_cfg['data'].get('past_frames', 5),
        past_stride=train_cfg['data'].get('past_stride', 5),
        past_jitter_std=train_cfg['data'].get('past_jitter_std', 0.0),
        past_jitter_prob=past_jitter_prob,
        return_clean_heatmap=use_clean_hm,
        use_mixed_stride=use_mixed_stride
    )
    val_ds = LeafTrackingDatasetV2(
        dataset_dir=train_cfg['data']['dataset_dir'],
        mode='val',
        img_size=tuple(train_cfg['data']['img_size']),
        past_window_minutes=train_cfg['data'].get('past_window_minutes', 30),
        past_frames=train_cfg['data'].get('past_frames', 5),
        past_stride=train_cfg['data'].get('past_stride', 5),
        past_jitter_std=train_cfg['data'].get('past_jitter_std', 0.0),
        past_jitter_prob=past_jitter_prob,
        allow_val_jitter=allow_val_jitter,
        return_clean_heatmap=use_clean_hm,
        use_mixed_stride=use_mixed_stride
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'],
        shuffle=True, 
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        prefetch_factor=1 if config['training'].get('num_workers', 4) > 0 else None,
        persistent_workers=(config['training'].get('num_workers', 4) > 0)
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['training']['batch_size'],
        shuffle=False, 
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        prefetch_factor=1 if config['training'].get('num_workers', 4) > 0 else None,
        persistent_workers=(config['training'].get('num_workers', 4) > 0)
    )
    
    # 3. Model
    print("Initializing Model V2...")
    
    w_conf_global = config['training'].get('w_conf', 1.0)
    use_conf = w_conf_global > 0
    
    if not use_conf:
        print("Disabling confidence head (w_conf=0).")
    
    model = DecoupledTracker(
        backbone=config['model']['backbone'], 
        pretrained=config['model']['pretrained'],
        past_frames=past_frames,
        dropout=config['model'].get('dropout', 0.1),
        use_conf=use_conf,
        multi_head=config['model'].get('multi_head', True)
    ).to(device)

    # Load weights if provided
    if args.weights:
        print(f"Loading weights from {args.weights} (strict=False)...")
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    
    # 4. Loss & Optimizer
    criterion = ImprovedLoss(
        w_box=config['training'].get('w_box', 5.0),
        w_kpt=config['training'].get('w_kpt', 20.0),
        w_conf=config['training'].get('w_conf', 1.0),
        w_temporal=config['training'].get('w_temporal', 0.0),
        w_attn=config['training'].get('w_attn', 10.0),
        use_wing=config['training'].get('use_wing', True)
    ).to(device)
    criterion.w_consistency = w_consistency_global
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config.get('optimizer', {}).get('weight_decay', 0.0)
    )
    
    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training'].get('scheduler_factor', 0.1),
        patience=config['training'].get('scheduler_patience', 5)
    )
    
    epochs = config['training']['epochs']
    freeze_epochs = config['training'].get('freeze_epochs', 5)
    
    early_stopping = EarlyStopping(patience=config['training']['patience'])
    
    start_epoch = 0
    results_file = save_dir / 'results.csv'
    if args.resume and results_file.exists():
        try:
            df = pd.read_csv(results_file)
            if len(df) > 0:
                start_epoch = int(df['epoch'].iloc[-1])
                print(f"Resuming from epoch {start_epoch}")
                
                # Automatically load weights from last.pt or best.pt if not provided
                if not args.weights:
                    # Priority: last.pt > epoch_{start_epoch}.pt > best.pt
                    w_paths = [
                        weights_dir / f'epoch_{start_epoch}.pt',
                        weights_dir / f'epoch_{start_epoch-1}.pt',
                        weights_dir / 'last.pt',
                        weights_dir / 'best.pt'
                    ]
                    for wp in w_paths:
                        if wp.exists():
                            print(f"Automatically loading weights for resume: {wp}")
                            model.load_state_dict(torch.load(wp, map_location=device))
                            break
                # Refresh plots
                plot_results(file=results_file)
        except Exception as e:
            print(f"Error reading results.csv for resume: {e}")
    
    if not results_file.exists():
        with open(results_file, 'w') as f:
            f.write('epoch,train_loss,val_loss,reward_err,base_err,tip_err,box_loss,kpt_loss,const_loss,loss_attn,val_box_loss,val_kpt_loss,val_const_loss,lr\n')

    # Reward-Based Early Stopping (flat config)
    reward_early_stopping = None
    if config['training'].get('use_reward_early_stopping', False):
        reward_early_stopping = RewardEarlyStopping(
            patience=config['training'].get('reward_patience', 5)
        )

    # Dropout parameters
    rgb_p = config['training'].get('rgb_dropout_p', 0.0)
    seq_p = config['training'].get('sequence_dropout_p', 0.0)

    # Override dataset jitter from training config
    train_jitter_std = config['training'].get('past_jitter_std', config['data'].get('past_jitter_std', 0.0))
    train_jitter_prob = config['training'].get('past_jitter_prob', config['data'].get('past_jitter_prob', 0.0))
    train_ds.past_jitter_std = train_jitter_std
    train_ds.past_jitter_prob = train_jitter_prob

    # Override loss weights from training config
    for weight_name in ['w_box', 'w_kpt', 'w_conf', 'w_temporal', 'w_consistency', 'w_attn']:
        if weight_name in config['training']:
            setattr(criterion, weight_name, config['training'][weight_name])

    min_epochs = config['training'].get('min_epochs', 0)

    for epoch in range(start_epoch, epochs):
        model.train()

        train_loss_accum = 0.0
        box_loss_accum = 0.0
        kpt_loss_accum = 0.0
        const_loss_accum = 0.0
        attn_loss_accum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            img = batch['image'].to(device) # [B, 3, H, W]
            hm = batch['heatmap'].to(device) # [B, 3, H, W]
            target = batch['target'].to(device) # [B, 8]
            
            B = img.shape[0]
            
            # Visualize First Batch
            if epoch == 0 and i == 0 and config['evaluation']['visualize']:
                plot_batch_input(img, target, fname=save_dir / 'train_batch.jpg')

            # Temporal Robustness Strategy (Mutually Exclusive)
            p_drop = random.random()
            
            if p_drop < seq_p:
                # 1. Sequence Dropout: No Past Information
                img_input = img
                hm_input = torch.zeros_like(hm)
                current_prev_conf = torch.zeros(B, 1).to(device)
            elif p_drop < (seq_p + rgb_p):
                # 2. RGB Dropout: No Current RGB (Blind Training)
                # Force reliance on Past Heatmap channels
                # Noise for uint8: Randomly centered around 127
                img_input = (torch.randn_like(img.float()) * 25.0 + 127.0).clamp(0, 255).to(torch.uint8)
                hm_input = hm
                current_prev_conf = torch.ones(B, 1).to(device) * 0.95
            else:
                # 3. Normal: Both available
                img_input = img
                hm_input = hm
                current_prev_conf = torch.ones(B, 1).to(device) * 0.95
            
            rgb_dropout_mode = config['training'].get('rgb_dropout_mode', 'noise')
            if rgb_dropout_mode == 'zeros' and p_drop >= seq_p and p_drop < (seq_p + rgb_p):
                img_input = torch.zeros_like(img)
                
            # Input Construction
            x = torch.cat([img_input, hm_input], dim=1) # [B, 3 + 15, H, W]
            
            # Feature-level Dropout probability
            f_drop_p = config['training'].get('feature_dropout_p', 0.3) if p_drop >= (seq_p + rgb_p) else 0.0

            optimizer.zero_grad()
            pred_reg, pred_conf, pred_masks = model(x, None, feature_dropout_p=f_drop_p)
            # Consistency Path (Clean)
            pred_reg_clean = None
            if 'heatmap_clean' in batch and criterion.w_consistency > 0 and p_drop >= seq_p:
                hm_clean = batch['heatmap_clean'].to(device)
                x_clean = torch.cat([img, hm_clean], dim=1)
                # Forward Clean Path without grad AND without dropout to act as a stable anchor
                with torch.no_grad():
                    pred_reg_clean, _, _ = model(x_clean, None, disable_dropout=True)

            # Forward pass
            temporal_target = batch['temporal_target'].to(device)
            loss, loss_dict = criterion(pred_reg, pred_conf, target, 
                                        masks=pred_masks,
                                        pred_reg_clean=pred_reg_clean,
                                        temporal_target=temporal_target,
                                        prev_conf_mask=current_prev_conf)
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss_accum += loss.item()
            box_loss_accum += loss_dict['loss_box']
            kpt_loss_accum += loss_dict['loss_kpt']
            
            pbar.set_postfix(loss_dict)
            
            const_loss_accum += loss_dict.get('loss_const', 0.0)
            attn_loss_accum += loss_dict.get('loss_attn', 0.0)
            
        avg_train_loss = train_loss_accum / len(train_loader)
        avg_box_loss = box_loss_accum / len(train_loader)
        avg_kpt_loss = kpt_loss_accum / len(train_loader)
        
        avg_const_loss = const_loss_accum / len(train_loader)
        avg_attn_loss = attn_loss_accum / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_box_loss_accum = 0.0
        val_kpt_loss_accum = 0.0
        val_const_loss_accum = 0.0
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                img = batch['image'].to(device)
                hm = batch['heatmap'].to(device)
                target = batch['target'].to(device)
                B = img.shape[0]
                
                prev_conf = torch.ones(B, 1).to(device)
                
                x = torch.cat([img, hm], dim=1)
                # Model returns (reg, conf, mask) 
                pred_reg, pred_conf, pred_masks = model(x, None)
                
                # Consistency Path (Clean) Validation
                pred_reg_clean = None
                if 'heatmap_clean' in batch and criterion.w_consistency > 0:
                    hm_clean = batch['heatmap_clean'].to(device)
                    x_clean = torch.cat([img, hm_clean], dim=1)
                    # model returns (reg, conf, mask) in eval mode
                    pred_reg_clean, _, _ = model(x_clean, None)

                temporal_target = batch['temporal_target'].to(device)
                loss, loss_dict = criterion(pred_reg, pred_conf, target, 
                                            masks=pred_masks,
                                            pred_reg_clean=pred_reg_clean,
                                            temporal_target=temporal_target)
                val_loss_accum += loss.item()
                val_box_loss_accum += loss_dict['loss_box']
                val_kpt_loss_accum += loss_dict['loss_kpt']
                val_const_loss_accum += loss_dict.get('loss_const', 0.0)
                
                # Visualize Predictions (Every Vis Freq)
                if i == 0 and (epoch % config['evaluation']['vis_freq'] == 0) and config['evaluation']['visualize']:
                     # Construct preds tensor from decoupled outputs
                    pred_box = pred_reg[:, :4]
                    pred_kpt = pred_reg[:, 4:]
                    # Combine [cx, cy, w, h, bx, by, tx, ty]
                    preds_combined = torch.cat([pred_box, pred_kpt], dim=1)
                    plot_batch_pred(img, target, preds_combined, fname=save_dir / f'val_batch_pred_{epoch}.jpg')
        
        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_box_loss = val_box_loss_accum / len(val_loader)
        avg_val_kpt_loss = val_kpt_loss_accum / len(val_loader)
        avg_val_const_loss = val_const_loss_accum / len(val_loader)
        
        # Scheduler Step
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Reward-Based Validation
        rew, b_err, t_err = 0.0, 0.0, 0.0
        reward_eval_freq = config['training'].get('reward_eval_freq', 5)
        if reward_early_stopping is not None:
            if (epoch + 1) % reward_eval_freq == 0:
                print(f"\n[REWARD] Performing recursive validation...")
                rew, b_err, t_err = perform_recursive_validation(model, device, config)
                print(f"[REWARD] Error: {rew:.2f} (Base: {b_err:.2f}, Tip: {t_err:.2f})")
                if reward_early_stopping(rew):
                    print("  [NEW BEST REWARD] Saving model...")
                    torch.save(model.state_dict(), weights_dir / 'best_reward.pt')
                if reward_early_stopping.early_stop and (epoch + 1) >= min_epochs:
                    print(f"Reward converged. Stopping training at epoch {epoch+1}.")
                    torch.save(model.state_dict(), weights_dir / 'last.pt')
                    break

        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Reward: {rew:.2f} | LR: {current_lr:.1e}")
        
        # Log
        with open(results_file, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss},{avg_val_loss},{rew},{b_err},{t_err},"
                    f"{avg_box_loss},{avg_kpt_loss},{avg_const_loss},{avg_attn_loss},"
                    f"{avg_val_box_loss},{avg_val_kpt_loss},{avg_val_const_loss},{current_lr}\n")
            
        # Plot Results
        plot_results(file=results_file)
            
        # Save Last & Best
        torch.save(model.state_dict(), weights_dir / 'last.pt')
        
        early_stopping(avg_val_loss)
        if early_stopping.best_loss == avg_val_loss:
             torch.save(model.state_dict(), weights_dir / 'best.pt')

        if (epoch + 1) % 5 == 0:
            save_attention_snapshot(model, device, save_dir, epoch + 1, dataset_dir=train_cfg['data']['dataset_dir'])
        if epoch + 1 >= 50 and early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
             
        # Cleanup memory at epoch end
        torch.cuda.empty_cache()
        import gc
        gc.collect()
            
        if epoch % 5 == 0:
            torch.save(model.state_dict(), weights_dir / f'epoch_{epoch}.pt')

    best_path = weights_dir / 'best.pt'
    if best_path.exists():
        model.load_state_dict(torch.load(best_path))
        run_comparison_eval(model, val_loader, device, save_dir)
        
        # --- Automated Recursive Evaluation ---
        print("\nRunning Automated Recursive Evaluation...")
        import subprocess
        import sys
        
        # Determine flags based on model configuration
        eval_cmd_base = [
            sys.executable, "scripts/evaluate_v3_4_recursive.py",
            "--weights", str(best_path),
            "--backbone", config['model']['backbone'],
            "--dataset_dir", config['data']['dataset_dir'],
            "--save_freq", "60" # Performance: Reduce video frames
        ]
        if config['model'].get('multi_head', True):
            eval_cmd_base.append("--multi_head")
        else:
            eval_cmd_base.append("--no_multi_head")
            
        if not use_conf:
            eval_cmd_base.append("--no_conf")
        if train_ds.use_mixed_stride:
            eval_cmd_base.append("--mixed_stride")
            
        # Area 34
        subprocess.run(eval_cmd_base + [
            "--output_subdir", f"{save_dir.name}_area34",
            "--leaf_ids", "0" # Representative leaf
        ])
        # Area 32 (Generalization check if exists/representative)
        subprocess.run(eval_cmd_base + [
            "--output_subdir", f"{save_dir.name}_area32",
            "--leaf_ids", "0" 
        ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--weights', type=str, default=None, help='Load weights from this path')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--start-stage', type=int, default=1, help='Stage to start from (1-indexed)')
    parser.add_argument('--exist-ok', action='store_true', help='Use existing directory without incrementing')
    parser.add_argument('--resume', action='store_true', help='Resume training from existing results.csv')
    parser.add_argument('--seed', type=int, default=None, help='Override global seed')
    args = parser.parse_args()
    train_v2(args)
