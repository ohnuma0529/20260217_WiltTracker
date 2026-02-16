
import torch
import torch.nn as nn
import torchvision.models as models

class HeatmapEncoder(nn.Module):
    """
    Lightweight CNN to encode past heatmaps into feature maps.
    Input: [B, 3*N, H, W] (uint8)
    Output: [B, 512, 7, 7]
    """
    def __init__(self, input_channels, output_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Simple residual-like blocks or just conv stack
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, output_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.encoder(x)

class SpatialAttentionFusionNoConf(nn.Module):
    """
    V2 Fusion: Generates 3 separate attention masks for BBox, Base, and Tip.
    Returns 3 sets of fused features and 3 masks.
    MODIFIED: Removed prev_conf_map input.
    """
    def __init__(self, dim_feats):
        super().__init__()
        # 3-channel spatial mask (BBox, Base, Tip)
        self.conv_mask = nn.Sequential(
            nn.Conv2d(dim_feats, dim_feats // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_feats // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_feats // 4, 3, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 3 independent fusion layers for each branch
        # Input dim is dim_feats * 2 (RGB + HM)
        # Removed + 1 for prev_conf
        fusion_input_dim = dim_feats * 2
        
        self.fusion_bbox = nn.Sequential(
            nn.Conv2d(fusion_input_dim, dim_feats, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_feats),
            nn.ReLU(inplace=True)
        )
        self.fusion_base = nn.Sequential(
            nn.Conv2d(fusion_input_dim, dim_feats, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_feats),
            nn.ReLU(inplace=True)
        )
        self.fusion_tip = nn.Sequential(
            nn.Conv2d(fusion_input_dim, dim_feats, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_feats),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_rgb, feat_hm):
        # 1. Generate 3 Spatial Masks
        masks = self.conv_mask(feat_hm) # [B, 3, 7, 7]
        m_bbox = masks[:, 0:1, :, :]
        m_base = masks[:, 1:2, :, :]
        m_tip = masks[:, 2:3, :, :]
        
        # 2. Apply Branch-specific Attention
        f_bbox_rgb = feat_rgb * (1.0 + m_bbox)
        f_base_rgb = feat_rgb * (1.0 + m_base)
        f_tip_rgb = feat_rgb * (1.0 + m_tip)
        
        # 3. Concatenate and Fuse for each branch
        # No prev_conf_map
        f_bbox = self.fusion_bbox(torch.cat([f_bbox_rgb, feat_hm], dim=1))
        f_base = self.fusion_base(torch.cat([f_base_rgb, feat_hm], dim=1))
        f_tip = self.fusion_tip(torch.cat([f_tip_rgb, feat_hm], dim=1))
        
        return (f_bbox, f_base, f_tip), masks

class DecoupledTrackerNoConf(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, past_frames=5, dropout=0.5, use_conf=False):
        super().__init__()
        # use_conf argument kept for signature compatibility but ignored for input logic
        self.use_conf_head = use_conf # But we can still have a head if we want? 
        # User said "conf入力とかはなし". Assuming input is gone. Output head typically depends on features.
        # If we remove input, we can still have output head.
        
        # 1. RGB Backbone
        self.backbone_name = backbone
        
        # 1. RGB Backbone & Dimensions
        if backbone == 'resnet18':
            base = models.resnet18(weights='DEFAULT' if pretrained else None)
            self.backbone_rgb = nn.Sequential(*list(base.children())[:-2])
            in_channels = 512
        elif backbone == 'resnet50':
            base = models.resnet50(weights='DEFAULT' if pretrained else None)
            self.backbone_rgb = nn.Sequential(*list(base.children())[:-2])
            in_channels = 2048
        elif backbone == 'efficientnet_b0':
            base = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
            self.backbone_rgb = base.features
            in_channels = 1280
        elif backbone == 'mobilenet_v3_large':
            base = models.mobilenet_v3_large(weights='DEFAULT' if pretrained else None)
            self.backbone_rgb = base.features
            in_channels = 960
        elif backbone == 'convnext_tiny':
            base = models.convnext_tiny(weights='DEFAULT' if pretrained else None)
            self.backbone_rgb = base.features
            in_channels = 768
        elif backbone == 'resnet101':
            base = models.resnet101(weights='DEFAULT' if pretrained else None)
            self.backbone_rgb = nn.Sequential(*list(base.children())[:-2])
            in_channels = 2048
        elif backbone == 'resnext101_32x8d':
            base = models.resnext101_32x8d(weights='DEFAULT' if pretrained else None)
            self.backbone_rgb = nn.Sequential(*list(base.children())[:-2])
            in_channels = 2048
        elif backbone == 'densenet121':
            base = models.densenet121(weights='DEFAULT' if pretrained else None)
            self.backbone_rgb = base.features
            in_channels = 1024
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")

        # Adapter to standardize feature dimension to 512
        self.dim_feats = 512
        if in_channels != self.dim_feats:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, self.dim_feats, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.dim_feats),
                nn.ReLU(inplace=True)
            )
        else:
            self.adapter = nn.Identity()
        
        # 2. Heatmap Encoder
        hm_channels = 3 * past_frames
        self.heatmap_encoder = HeatmapEncoder(hm_channels, output_dim=self.dim_feats)
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 3. Neck: Multi-Head Spatial Attention Fusion (No Conf)
        self.fusion = SpatialAttentionFusionNoConf(self.dim_feats)
        
        # 4. Heads (Perfectly Decoupled)
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden_dim = 256
        
        # BBox Head (Uses BBox branch)
        self.bbox_head = nn.Sequential(
            nn.Linear(self.dim_feats, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid() 
        )
        
        # Base Head (Uses Base branch)
        self.base_head = nn.Sequential(
            nn.Linear(self.dim_feats, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )
        
        # Tip Head (Uses Tip branch)
        self.tip_head = nn.Sequential(
            nn.Linear(self.dim_feats, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )
        
        if self.use_conf_head:
            # Confidence Head (Restored for Evaluator compatibility)
            self.conf_head = nn.Sequential(
                nn.Linear(self.dim_feats, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1)
            )

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, prev_conf=None, feature_dropout_p=0.0, disable_dropout=False):
        # NOTE: prev_conf arg is kept for compatibility with training loop signature, 
        # but it is NOT used in logic.
        
        # Split and Normalize
        x_rgb_raw = x[:, :3, :, :].float() / 255.0
        x_hm = x[:, 3:, :, :].float() / 255.0
        x_rgb = (x_rgb_raw - self.mean) / self.std

        # Feature Extraction
        # Feature Extraction
        feat_rgb_raw = self.backbone_rgb(x_rgb)
        feat_rgb = self.adapter(feat_rgb_raw)
        feat_hm = self.heatmap_encoder(x_hm)
        
        if self.training and feature_dropout_p > 0:
            mask = (torch.rand(feat_rgb.shape[0], 1, 1, 1, device=feat_rgb.device) > feature_dropout_p).float()
            feat_rgb = feat_rgb * mask
            
        # Fusion (No prev_conf passed)
        (f_bbox, f_base, f_tip), masks = self.fusion(feat_rgb, feat_hm)
            
        # Global Pooling and Heads
        gap_bbox = self.gap(f_bbox).flatten(1)
        gap_base = self.gap(f_base).flatten(1)
        gap_tip = self.gap(f_tip).flatten(1)
        
        if not disable_dropout:
            gap_bbox = self.dropout_layer(gap_bbox)
            gap_base = self.dropout_layer(gap_base)
            gap_tip = self.dropout_layer(gap_tip)

        pred_bbox = self.bbox_head(gap_bbox) # [B, 4]
        pred_base = self.base_head(gap_base) # [B, 2]
        pred_tip = self.tip_head(gap_tip)   # [B, 2]
        
        pred_reg = torch.cat([pred_bbox, pred_base, pred_tip], dim=1) # [B, 8]
        
        pred_conf = None
        if self.use_conf_head:
            pred_conf = self.conf_head(gap_bbox) # [B, 1]
        
        if self.training:
            return pred_reg, pred_conf
        else:
            return pred_reg, pred_conf, masks
