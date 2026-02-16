
import os
import glob
import re
import random
from datetime import datetime
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors
from PIL import Image
import numpy as np

class LeafTrackingDatasetV2(Dataset):
    def __init__(self, dataset_dir, mode='train', img_size=(224, 224), 
                 past_window_minutes=30, past_frames=5, past_stride=5,
                 past_jitter_std=0.0, past_jitter_prob=1.0, 
                 allow_val_jitter=False, return_clean_heatmap=False,
                 use_mixed_stride=False):
        """
        Args:
            dataset_dir: Path to 'dataset_v2' containing 'images' and 'labels'.
            mode: 'train' or 'val'.
            img_size: Target size (H, W).
            past_window_minutes: Max time difference to consider for 'past' frame.
            past_frames: Number of past frames to use as input.
        """
        self.images_dir = os.path.join(dataset_dir, 'images')
        self.labels_dir = os.path.join(dataset_dir, 'labels')
        self.mode = mode
        self.img_size = img_size
        self.past_window_minutes = past_window_minutes
        self.past_frames = past_frames
        self.past_stride = past_stride
        self.past_jitter_std = past_jitter_std
        self.past_jitter_prob = past_jitter_prob
        self.allow_val_jitter = allow_val_jitter
        self.return_clean_heatmap = return_clean_heatmap
        self.use_mixed_stride = use_mixed_stride
        
        # 1. Load Data Map
        self.image_bases, self.labels, self.past_indices, self.valid_past_mask = self._load_samples()
        
        # 2. Pre-calculate Gaussian Kernel to avoid repeated allocations
        self.sigma = 3
        self.radius = self.sigma * 3
        side = 2 * self.radius + 1
        y, x = torch.meshgrid(torch.arange(side) - self.radius, torch.arange(side) - self.radius, indexing='ij')
        self.kernel_scaled = (torch.exp(-(x**2 + y**2) / (2 * self.sigma**2)) * 255).to(torch.uint8)

        # 3. Define Augmentation
        if mode == 'train':
            self.transforms = v2.Compose([
                v2.RandomApply([v2.RandomResizedCrop(img_size, scale=(0.8, 1.0))], p=0.3),
                v2.RandomApply([v2.RandomAffine(degrees=10, translate=(0.05, 0.05))], p=0.3),
                v2.Resize(img_size),
            ])
        else:
            self.transforms = v2.Compose([
                v2.Resize(img_size),
            ])

    def _load_samples(self):
        # ... (Group logic remains same, but we optimize storage)
        all_labels = sorted(glob.glob(os.path.join(self.labels_dir, "*.txt")))
        has_val_date = any('20251114' in f for f in all_labels)
        
        filtered_files = []
        for lf in all_labels:
            if has_val_date:
                if self.mode == 'train':
                    if '20251114' not in lf: filtered_files.append(lf)
                elif self.mode == 'val':
                    if '20251114' in lf: filtered_files.append(lf)
                else:
                    filtered_files.append(lf)
            else:
                filtered_files.append(lf)

        records_list = []
        for lf in filtered_files:
            base = os.path.basename(lf).replace('.txt', '')
            parts = base.split('_')
            if len(parts) != 4: continue
            area_id, leaf_id, date_str, ts_str = parts
            key = f"{area_id}_{leaf_id}_{date_str}"
            try:
                dt = datetime.strptime(f"{date_str}-{ts_str}", "%Y%m%d-%H%M")
            except: continue
            bbox, kpts = self._read_label(lf)
            if bbox is None: continue
            records_list.append({
                'key': key, 'ts': dt, 'base': base, 
                'label': np.array(bbox + kpts, dtype=np.float32)
            })

        records_list.sort(key=lambda x: x['ts'])
        grouped_indices = {}
        for idx, rec in enumerate(records_list):
            key = rec['key']
            if key not in grouped_indices: grouped_indices[key] = []
            grouped_indices[key].append(idx)

        num_samples = len(records_list)
        # Use fixed-length numpy array for bases to avoid CoW on Python string objects
        image_bases = np.array([rec['base'] for rec in records_list], dtype='S64')
        labels = np.stack([rec['label'] for rec in records_list])
        past_indices = np.zeros((num_samples, self.past_frames), dtype=np.int32)
        valid_past_mask = np.zeros((num_samples, self.past_frames), dtype=bool)

        for i in range(num_samples):
            curr = records_list[i]
            group = grouped_indices[curr['key']]
            curr_pos = group.index(i)
            p_count = 0
            
            # Use past_stride to pick frames at intervals
            # If use_mixed_stride=True, first frame is t-1, others are t-k*stride
            for p_num in range(1, self.past_frames + 1):
                if self.use_mixed_stride:
                    # Mixed stride: [1, stride, 2*stride, 3*stride, 4*stride]
                    k_stride = 1 if p_num == 1 else (p_num - 1) * self.past_stride
                    check_pos = curr_pos - k_stride
                else:
                    # Standard stride: [1*stride, 2*stride, 3*stride, ...]
                    check_pos = curr_pos - (p_num * self.past_stride)
                
                if check_pos < 0: break
                
                prev_idx = group[check_pos]
                prev = records_list[prev_idx]
                diff = (curr['ts'] - prev['ts']).total_seconds() / 60.0
                
                if diff <= self.past_window_minutes:
                    past_indices[i, p_count] = prev_idx
                    valid_past_mask[i, p_count] = True
                    p_count += 1
                else:
                    break
        return image_bases, labels, past_indices, valid_past_mask

    def __len__(self):
        return len(self.image_bases)

    def _preload_images_to_ram(self):
        # Deprecated logic removed to keep it clean. 
        # Caching now uses shared torch.Tensor only if explicitly needed, but we avoid it here.
        pass

    def _read_label(self, txt_path):
        """
        Reads 0 cx cy w h bx by tx ty
        Returns: bbox (cx cy w h), kpts (bx by, tx ty)
        Coordinates are normalized relative to image size.
        """
        if not os.path.exists(txt_path):
            return None, None
            
        with open(txt_path, 'r') as f:
            line = f.readline().strip()
            if not line: return None, None
            parts = list(map(float, line.split()))
            # 0, cx, cy, w, h, bx, by, tx, ty
            
            bbox = parts[1:5]
            kpts = [parts[5], parts[6], parts[7], parts[8]] # bx, by, tx, ty
            
            return bbox, kpts

    def _generate_heatmap(self, kpts, size):
        H, W = size
        heatmap = torch.zeros(3, H, W, dtype=torch.uint8)
        
        for i in range(3):
            cx, cy = int(kpts[i][0].item()), int(kpts[i][1].item())
            if cx < 0 or cx >= W or cy < 0 or cy >= H: continue
            
            x1, x2 = max(0, cx - self.radius), min(W, cx + self.radius + 1)
            y1, y2 = max(0, cy - self.radius), min(H, cy + self.radius + 1)
            kx1, kx2 = x1 - (cx - self.radius), x2 - (cx - self.radius)
            ky1, ky2 = y1 - (cy - self.radius), y2 - (cy - self.radius)
            heatmap[i, y1:y2, x1:x2] = torch.maximum(heatmap[i, y1:y2, x1:x2], self.kernel_scaled[ky1:ky2, kx1:kx2])
        return heatmap

    def __getitem__(self, idx):
        # 1. Load Current Image
        base = self.image_bases[idx].decode('utf-8')
        img_path = os.path.join(self.images_dir, base + '.jpg')
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except:
            img_pil = Image.new('RGB', (224, 224))
            
        W, H = img_pil.size
        
        label_vec = self.labels[idx] # [cx, cy, w, h, bx, by, tx, ty]
        curr_bbox = label_vec[0:4]
        curr_kpts = label_vec[4:8]

        # Convert to TV Tensors
        tv_img = tv_tensors.Image(img_pil)
        
        # BBox: (Normalized) -> (Absolute Pixels)
        curr_bbox_abs = [
            curr_bbox[0] * W,
            curr_bbox[1] * H,
            curr_bbox[2] * W,
            curr_bbox[3] * H
        ]

        tv_bbox = tv_tensors.BoundingBoxes(
            [curr_bbox_abs], 
            format=tv_tensors.BoundingBoxFormat.CXCYWH, 
            canvas_size=(H, W)
        )
        
        # Keypoints: Collect All (Current + N * Past)
        # Current: [Base, Tip] (2 pts)
        curr_points = [
            [curr_kpts[0] * W, curr_kpts[1] * H], 
            [curr_kpts[2] * W, curr_kpts[3] * H]
        ]
        
        past_points_flat_clean = []
        past_points_flat_noisy = []
        valid_mask = [] # True if past exists
        
        # Store clean past keypoints for potential clean heatmap generation
        clean_past_kpts_for_heatmap = []

        for i in range(self.past_frames):
            if self.valid_past_mask[idx, i]:
                p_idx = self.past_indices[idx, i]
                p_label = self.labels[p_idx]
                # [cx, cy, w, h, bx, by, tx, ty]
                pcx, pcy, _, _ = p_label[0:4]
                pbx, pby, ptx, pty = p_label[4:8]
                
                # Clean copy (normalized)
                past_points_flat_clean.extend([
                    [pcx * W, pcy * H],
                    [pbx * W, pby * H],
                    [ptx * W, pty * H]
                ])

                # Store clean keypoints for heatmap generation (normalized, then converted to pixel space later)
                clean_past_kpts_for_heatmap.append([
                    [pcx, pcy],
                    [pbx, pby],
                    [ptx, pty]
                ])

                # Apply jitter if conditions met
                if (self.mode == 'train' or self.allow_val_jitter) and self.past_jitter_std > 0 and random.random() < self.past_jitter_prob:
                    # Apply jitter in pixel space (sigma is in pixels)
                    # W, H are available here from img_pil.size
                    pcx += np.random.normal(0, self.past_jitter_std) / W
                    pcy += np.random.normal(0, self.past_jitter_std) / H
                    pbx += np.random.normal(0, self.past_jitter_std) / W
                    pby += np.random.normal(0, self.past_jitter_std) / H
                    ptx += np.random.normal(0, self.past_jitter_std) / W
                    pty += np.random.normal(0, self.past_jitter_std) / H

                # Noisy copy for heatmap input
                past_points_flat_noisy.extend([
                    [pcx * W, pcy * H],
                    [pbx * W, pby * H],
                    [ptx * W, pty * H]
                ])
                valid_mask.append(True)
            else:
                past_points_flat_clean.extend([[0,0], [0,0], [0,0]])
                past_points_flat_noisy.extend([[0,0], [0,0], [0,0]])
                valid_mask.append(False)
                
        # Total points: (2 + 3 * N) * 2 sets
        all_points = curr_points + past_points_flat_clean + past_points_flat_noisy
        
        tv_kpts = tv_tensors.KeyPoints(
            all_points, 
            canvas_size=(H, W)
        )
        
        # APPLY TRANSFORMS
        out_img, out_bbox, out_kpts = self.transforms(tv_img, tv_bbox, tv_kpts)
        
        # 1. Target Regression (Current)
        # BBox
        obox = out_bbox[0]
        ncx = obox[0] / self.img_size[1]
        ncy = obox[1] / self.img_size[0]
        nw  = obox[2] / self.img_size[1]
        nh  = obox[3] / self.img_size[0]
        
        # Keypoints (Current)
        nkpts = out_kpts.clone().float()
        nkpts[:, 0] /= self.img_size[1]
        nkpts[:, 1] /= self.img_size[0]
        
        num_pts_per_set = 2 + (self.past_frames * 3)
        nkpts_clean = nkpts[:num_pts_per_set]
        nkpts_noisy = nkpts[num_pts_per_set:]
        
        target_reg = torch.tensor([ncx, ncy, nw, nh, nkpts_clean[0,0], nkpts_clean[0,1], nkpts_clean[1,0], nkpts_clean[1,1]], dtype=torch.float32)
        
        heatmaps_list = []
        clean_heatmaps_list = [] if self.return_clean_heatmap else None

        for i in range(self.past_frames):
            # Noisy points start at num_pts_per_set, and there are 3 per frame
            noisy_start_idx = num_pts_per_set + (i * 3)
            # Clean past points start at index 2
            clean_start_idx = 2 + (i * 3)
            
            if valid_mask[i]:
                # Slice [Center, Base, Tip] from noisy set for heatmap input
                p_kpts_set = out_kpts[noisy_start_idx : noisy_start_idx + 3]
                hm = self._generate_heatmap(p_kpts_set, self.img_size) # [3, H, W] uint8
                
                if self.return_clean_heatmap:
                    p_kpts_clean_set = out_kpts[clean_start_idx : clean_start_idx + 3]
                    hm_c = self._generate_heatmap(p_kpts_clean_set, self.img_size)
                    clean_heatmaps_list.append(hm_c)
            else:
                hm = torch.zeros(3, self.img_size[0], self.img_size[1], dtype=torch.uint8)
                if self.return_clean_heatmap:
                    clean_heatmaps_list.append(hm.clone())
            
            heatmaps_list.append(hm)
            
        # Stack all heatmaps: [3*N, H, W]
        if heatmaps_list:
            stacked_heatmap = torch.cat(heatmaps_list, dim=0)
        else:
            stacked_heatmap = torch.zeros(0, self.img_size[0], self.img_size[1], dtype=torch.uint8) 
            

        # Has ANY past?
        has_any_past = any(valid_mask)

        # 3. Create temporal Smoothness Target
        # Use first valid past frame as target for smoothness if it exists
        temporal_target = target_reg.clone() # Default to self if no past
        if has_any_past:
            # Find closest valid past frame index
            first_valid_idx = -1
            for i in range(self.past_frames):
                if valid_mask[i]:
                    first_valid_idx = i
                    break
            
            if first_valid_idx != -1:
                p_start = 2 + (first_valid_idx * 3)
                p_pts = nkpts_clean[p_start : p_start+3] # [pc, pb, pt] (Use Clean Set!)
                # Construct regression vector for past: [pc_x, pc_y, 0, 0, pb_x, pb_y, pt_x, pt_y] 
                # (w/h doesn't matter much for smoothness, keep current or zero)
                temporal_target = torch.tensor([
                    p_pts[0,0], p_pts[0,1], nw, nh, 
                    p_pts[1,0], p_pts[1,1], p_pts[2,0], p_pts[2,1]
                ], dtype=torch.float32)

        sample = {
            'image': out_img.to(torch.uint8), # Standardize to uint8
            'heatmap': stacked_heatmap,      # uint8
            'target': target_reg,            # float32
            'temporal_target': temporal_target, # float32
            'has_past': has_any_past         # bool
        }
        if self.return_clean_heatmap:
            sample['heatmap_clean'] = torch.cat(clean_heatmaps_list, dim=0)
            
        return sample
