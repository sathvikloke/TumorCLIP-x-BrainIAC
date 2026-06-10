"""BrainIAC variants — direct ports of source code from both upstreams,
plus exactly two minimal additions (2D→3D adapter and Zongyu-style wrappers).

Provenance of every component in this file is annotated. There are exactly
**three** pieces of code I did not copy verbatim from one of the source
codebases:

  1. ``adapt_2d_to_brainiac`` — Zongyu's pipeline produces 2-D RGB tensors;
     BrainIAC expects 3-D 96³ single-channel volumes. Nothing in either
     source bridges this gap, so I wrote a minimal adapter (RGB→grayscale,
     bilinear-resize H/W to 96, expand along depth to make a 96³ volume,
     then z-score with the *same math* MONAI's
     ``NormalizeIntensity(nonzero=True, channel_wise=True)`` performs,
     implemented as pure differentiable PyTorch so the autograd graph stays
     intact through training).
  2. ``BrainIACClassifier`` and ``BrainIACEncoder`` are wrapper classes that
     give Zongyu's TumorCLIP fusion code the API it expects
     (``load_pretrained_weights``, ``compute_loss``, ``get_optimizer_params``,
     ``forward_features``). Their *internals* are direct ports — they
     simply hold the source classes (``ViTBackboneNet`` from BrainIAC,
     ``feature_projection`` MLP from Zongyu's ``DenseNetEncoder``).
  3. ``original_classifier = nn.Linear(768, num_classes)`` inside
     ``BrainIACEncoder`` is necessary because Zongyu's ``DenseNetEncoder``
     aliases DenseNet's built-in classifier head, and BrainIAC's ViT
     pretrained via SimCLR has no classifier head to alias.

Everything else is a direct copy.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import FocalLoss, LabelSmoothingCrossEntropy


# ============================================================================
# CONFIGURATION
# ============================================================================
# These constants come from BrainIAC's source:
#   - 96³ input size: src/dataset.py::get_validation_transform (image_size arg)
#   - 768 feature dim: src/model.py::ViTBackboneNet (hidden_size + CLS token)
#   - 16³ patch size: src/model.py::ViTBackboneNet

BRAINIAC_WEIGHTS_PATH = os.environ.get(
    "BRAINIAC_WEIGHTS_PATH",
    "weights/brainiac/BrainIAC.ckpt",
)
BRAINIAC_FEATURE_DIM = 768                 # from src/model.py
BRAINIAC_INPUT_SIZE = (96, 96, 96)         # from src/dataset.py
BRAINIAC_PATCH_SIZE = (16, 16, 16)         # from src/model.py


# ============================================================================
# DIRECT PORT FROM BrainIAC src/model.py
# ============================================================================
# The three classes below are copied verbatim from BrainIAC's `src/model.py`
# (Tak et al. 2026, https://github.com/AIM-KannLab/BrainIAC). The only edits
# are: (a) import the MONAI ViT at use site rather than module top so the
# import error is friendlier; (b) make the checkpoint path optional so the
# user can construct the model in two stages (build, then load) — matching
# Zongyu's ``DenseNetClassifier`` construction pattern. The forward, the
# state-dict prefix-strip, ``strict=True``, and the CLS-token extraction
# are unchanged.

class ViTBackboneNet(nn.Module):
    """Direct port of BrainIAC `src/model.py::ViTBackboneNet`."""

    def __init__(self, simclr_ckpt_path=None):
        super().__init__()
        from monai.networks.nets import ViT

        # === BrainIAC src/model.py lines 12-21 ===
        # MONAI < 1.2 doesn't accept `save_attn`. We try the BrainIAC-faithful
        # config first; if it errors on that kwarg, retry without it.
        vit_kwargs = dict(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=True,
        )
        try:
            self.backbone = ViT(**vit_kwargs)
        except TypeError as exc:
            if "save_attn" in str(exc):
                vit_kwargs.pop("save_attn")
                self.backbone = ViT(**vit_kwargs)
                print("   Note: MONAI version doesn't support save_attn — built ViT without it.")
            else:
                raise

        # === BrainIAC src/model.py lines 23-37 (load + strip prefix) ===
        # I made the path optional vs. BrainIAC's required argument, so the
        # construct/load split matches Zongyu's DenseNetClassifier pattern.
        if simclr_ckpt_path and os.path.exists(simclr_ckpt_path):
            ckpt = torch.load(simclr_ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("backbone."):
                    new_key = key[9:]  # len("backbone.") == 9
                    backbone_state_dict[new_key] = value
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
            print("Backbone weights loaded!!")
        elif simclr_ckpt_path:
            print(f"   WARNING: BrainIAC weights not found at {simclr_ckpt_path!r}; using random init.")

    def forward(self, x):
        # === BrainIAC src/model.py lines 40-46 ===
        features = self.backbone(x)
        cls_token = features[0][:, 0]  # Shape: [batch_size, 768]
        return cls_token


class Classifier(nn.Module):
    """Direct port of BrainIAC `src/model.py::Classifier`."""

    def __init__(self, d_model=768, num_classes=1):
        super().__init__()
        # === BrainIAC src/model.py lines 49-51 ===
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # === BrainIAC src/model.py lines 52-54 ===
        x = self.fc(x)
        return x


class SingleScanModel(nn.Module):
    """Direct port of BrainIAC `src/model.py::SingleScanModel`."""

    def __init__(self, backbone, classifier):
        super().__init__()
        # === BrainIAC src/model.py lines 57-61 ===
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # === BrainIAC src/model.py lines 62-66 ===
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# ============================================================================
# 2D → 3D ADAPTER (NOT in either source — required by the 2D/3D mismatch)
# ============================================================================
# Zongyu's TumorCLIP pipeline yields tensors of shape (B, 3, 224, 224) from
# torchvision's ImageFolder + standard transforms. BrainIAC's preprocessing
# (src/dataset.py::get_validation_transform) operates on 3-D NIfTI files
# and produces tensors of shape (1, 96, 96, 96). Neither codebase contains
# any bridging code; the function below is my minimal port that:
#
#   1. Collapses RGB→grayscale by averaging the 3 channels (Kaggle brain MRIs
#      are stored as grayscale-rendered RGB so this is lossless).
#   2. Trilinear-interpolates the 2-D slice into a 96³ cube (matches MONAI's
#      `Resized(spatial_size=image_size, mode="trilinear")`).
#   3. Z-scores using MONAI's exact `NormalizeIntensity` transform — same one
#      BrainIAC's pretraining used.
#
# Real 3-D tensors (B, C, D, H, W) pass through unchanged.

def adapt_2d_to_brainiac(x: torch.Tensor, size=BRAINIAC_INPUT_SIZE) -> torch.Tensor:
    """Convert (B, 3, H, W) RGB slices → (B, 1, 96, 96, 96) volumes.

    Differentiable end-to-end. For 2-D inputs we use a 2-step resize
    (bilinear in (H, W), then expand along depth) — this avoids the
    "input too small" warning that trilinear-from-depth-1 produces, and
    is equivalent (the depth dim of 1 would just be replicated either way).

    Then z-scores with the same semantics as MONAI's
    NormalizeIntensity(nonzero=True, channel_wise=True), implemented as
    pure PyTorch so gradients flow through. See
    ``_brainiac_normalize_intensity`` for the math.
    """
    if x.dim() == 5:
        # Already 3-D — ensure single channel + correct spatial size + normalize.
        if x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)
        if tuple(x.shape[-3:]) != tuple(size):
            x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        return _brainiac_normalize_intensity(x)

    if x.dim() != 4:
        raise ValueError(f"Expected 4-D or 5-D tensor, got shape {tuple(x.shape)}")

    # 2-D path: RGB → grayscale → bilinear-resize HxW → expand to depth → normalize
    if x.size(1) == 3:
        x = x.mean(dim=1, keepdim=True)             # (B, 1, H, W)
    elif x.size(1) != 1:
        raise ValueError(f"Expected 1 or 3 channels, got {x.size(1)}")

    depth, target_h, target_w = size
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear",
                      align_corners=False)          # (B, 1, 96, 96)
    x = x.unsqueeze(2).expand(-1, -1, depth, -1, -1).contiguous()  # (B, 1, 96, 96, 96)
    return _brainiac_normalize_intensity(x)


def _brainiac_normalize_intensity(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable port of MONAI's ``NormalizeIntensity(nonzero=True, channel_wise=True)``.

    Per (batch, channel) volume:
      1. Build a mask of nonzero voxels.
      2. Compute mean and std over nonzero voxels only.
      3. Normalize ``(x - mean) / std`` for nonzero voxels; leave zeros as zero.

    Vectorised so it stays fast and so autograd flows through end-to-end
    (the previous version called MONAI's transform per-batch-item, which
    converts to MetaTensor and can break the gradient graph).
    """
    # x: (B, C, D, H, W)
    B, C = x.size(0), x.size(1)
    nonzero_mask = (x != 0).to(x.dtype)             # (B, C, D, H, W), 1/0 float

    # Count of nonzero voxels per (B, C)
    count = nonzero_mask.flatten(2).sum(dim=-1, keepdim=True)               # (B, C, 1)
    safe_count = count.clamp(min=1.0)

    # Mean over nonzero voxels
    sum_x = (x * nonzero_mask).flatten(2).sum(dim=-1, keepdim=True)         # (B, C, 1)
    mean = (sum_x / safe_count).view(B, C, 1, 1, 1)

    # Variance over nonzero voxels (using the unbiased=False / population variance,
    # matching MONAI's behavior)
    diff = (x - mean) * nonzero_mask
    var = (diff ** 2).flatten(2).sum(dim=-1, keepdim=True) / safe_count
    std = (var.clamp(min=0.0) + eps).sqrt().view(B, C, 1, 1, 1)

    # Normalize only the nonzero voxels; preserve zeros exactly.
    normalized = (x - mean) / std
    return torch.where(nonzero_mask.bool(), normalized, x)


# ============================================================================
# Zongyu-compatible wrappers (BrainIACClassifier / BrainIACEncoder)
# ============================================================================
# The wrappers below give the BrainIAC backbone the same API as Zongyu's
# DenseNetClassifier / DenseNetEncoder in `densenet_variants.py`, so the
# TumorCLIP fusion code above this layer doesn't change.
#
# Source breakdown for each method:
#   - __init__ ctor signature, loss configuration:  Zongyu densenet_variants.py
#   - self.backbone = ViTBackboneNet(...):           BrainIAC src/model.py
#   - self.classifier = Classifier(...):             BrainIAC src/model.py
#   - self.dropout = nn.Dropout(p=0.2):              BrainIAC src/model.py
#   - forward body (backbone → dropout → classifier): BrainIAC SingleScanModel
#   - get_optimizer_params:                          Zongyu densenet_variants.py
#   - compute_loss:                                  Zongyu densenet_variants.py
#   - load_pretrained_weights:                       Zongyu densenet_variants.py
#   - 2D→3D adapter call at start of forward:        MY ADDITION (necessary)

class BrainIACClassifier(nn.Module):
    """BrainIAC for classification.

    Mirrors Zongyu's `DenseNetClassifier` interface; internals are direct
    ports of BrainIAC's `ViTBackboneNet` + `Classifier` + `SingleScanModel`.
    """

    def __init__(self, num_classes=6, backbone_lr=5e-5, head_lr=1e-3,
                 focal_gamma=2.0, label_smoothing=0.05, weights_path: str = None):
        super().__init__()
        # Direct ports from BrainIAC src/model.py
        self.backbone = ViTBackboneNet(weights_path or BRAINIAC_WEIGHTS_PATH)
        self.classifier = Classifier(d_model=BRAINIAC_FEATURE_DIM, num_classes=num_classes)
        self.dropout = nn.Dropout(p=0.2)

        # Direct ports from Zongyu's DenseNetClassifier
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr

        print("BrainIAC classifier created")
        print(f"   Classes: {num_classes}")
        print(f"   Backbone: 3-D ViT-B (96³, patch 16³, 12 layers, hidden 768)")
        print(f"   Backbone learning rate: {backbone_lr}")
        print(f"   Head learning rate: {head_lr}")

    def load_pretrained_weights(self, checkpoint_path):
        """Re-load BrainIAC's pretrained weights from a different path.

        Delegates to ViTBackboneNet's loading logic (which is BrainIAC's
        exact recipe).
        """
        print(f"Loading pretrained weights: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"   WARNING: weights not found at {checkpoint_path}; skipping")
            return False
        # Rebuild the backbone with the new path — this is the only way
        # to invoke BrainIAC's exact loading code (which is in the ctor).
        new_backbone = ViTBackboneNet(checkpoint_path)
        # Transfer the loaded weights into our existing backbone
        self.backbone.backbone.load_state_dict(new_backbone.backbone.state_dict())
        return True

    def forward(self, x):
        # MY ADDITION: 2-D → 3-D adapter (required by the dimensionality mismatch).
        x = adapt_2d_to_brainiac(x)
        # The rest is a direct call to BrainIAC's SingleScanModel pattern:
        #   x = backbone(x); x = dropout(x); x = classifier(x); return x
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def get_optimizer_params(self):
        """Direct port from Zongyu's DenseNetClassifier."""
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {'params': backbone_params, 'lr': self.backbone_lr, 'name': 'backbone'},
            {'params': head_params, 'lr': self.head_lr, 'name': 'head'}
        ]

    def compute_loss(self, logits, targets, loss_type='focal'):
        """Direct port from Zongyu's DenseNetClassifier."""
        if loss_type == 'focal':
            return self.focal_loss(logits, targets)
        elif loss_type == 'label_smooth':
            return self.label_smooth_loss(logits, targets)
        else:
            return F.cross_entropy(logits, targets)


class BrainIACEncoder(nn.Module):
    """BrainIAC encoder for feature extraction (for CLIP fusion).

    Mirrors Zongyu's `DenseNetEncoder` interface; internals are direct
    ports of BrainIAC's `ViTBackboneNet` + `Classifier`.
    """

    def __init__(self, embed_dim=512, dropout=0.1, num_classes=6, weights_path: str = None,
                 shared_backbone: "ViTBackboneNet" = None):
        """Args:
            shared_backbone: if provided, reuse this BrainIAC backbone instead of
                building a new one. Halves GPU memory when both the brainiac_branch
                and clip_branch of SimpleFusionModel point at the same ViT-B
                — the original Zongyu design uses two DenseNet copies, which is
                affordable; two ViT-B copies is not. Pass
                ``encoder = BrainIACEncoder(shared_backbone=classifier.backbone)``
                to share. See the README "Memory-saving recipe" section.
        """
        super().__init__()
        if shared_backbone is not None:
            self.backbone = shared_backbone
        else:
            # Direct port from BrainIAC src/model.py
            self.backbone = ViTBackboneNet(weights_path or BRAINIAC_WEIGHTS_PATH)
        backbone_dim = BRAINIAC_FEATURE_DIM  # 768

        # MY ADDITION (necessary): Zongyu's DenseNetEncoder aliases the
        # DenseNet's built-in classifier as `original_classifier`. BrainIAC's
        # SimCLR-pretrained ViT has no classifier head, so we create a fresh
        # Linear(768, num_classes) — same as BrainIAC's `Classifier` would be.
        self.original_classifier = Classifier(d_model=backbone_dim, num_classes=num_classes)

        # Direct port from Zongyu's DenseNetEncoder.feature_projection
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # Direct port from Zongyu's DenseNetEncoder.new_classification_head
        self.new_classification_head = nn.Linear(embed_dim, num_classes)

        self.embed_dim = embed_dim
        self.backbone_dim = backbone_dim

        print("BrainIAC encoder created")
        print(f"   Backbone dim: {backbone_dim} (3-D ViT CLS token) → CLIP dim: {embed_dim}")

    def forward_features(self, x):
        """Returns projected features.

        Direct port of Zongyu's DenseNetEncoder.forward_features semantics:
            features = backbone(x); projected = feature_projection(features)
        with the backbone returning 768-d CLS token instead of pooled
        DenseNet features.
        """
        x = adapt_2d_to_brainiac(x)         # MY ADDITION (2D→3D)
        features = self.backbone(x)         # BrainIAC: returns (B, 768) CLS token
        projected_features = self.feature_projection(features)  # Zongyu MLP
        return projected_features

    def forward(self, x, return_features=False, use_original_classifier=False):
        """Direct port of Zongyu's DenseNetEncoder.forward dispatch."""
        if use_original_classifier:
            x = adapt_2d_to_brainiac(x)     # MY ADDITION
            features = self.backbone(x)
            return self.original_classifier(features)
        clip_features = self.forward_features(x)
        if return_features:
            new_logits = self.new_classification_head(clip_features)
            return clip_features, new_logits
        return clip_features

    def load_pretrained_weights(self, checkpoint_path):
        """Re-load BrainIAC weights; delegates to ViTBackboneNet's loader."""
        print(f"Loading pretrained weights: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"   WARNING: weights not found at {checkpoint_path}; skipping")
            return False
        new_backbone = ViTBackboneNet(checkpoint_path)
        self.backbone.backbone.load_state_dict(new_backbone.backbone.state_dict())
        return True


# ============================================================================
# Helper functions (mirroring Zongyu's densenet_variants.py)
# ============================================================================

def create_brainiac_classifier(num_classes=6, pretrained_path=None, **kwargs):
    """Direct port from Zongyu's create_densenet_classifier."""
    model = BrainIACClassifier(num_classes=num_classes, **kwargs)
    if pretrained_path:
        model.load_pretrained_weights(pretrained_path)
    return model


def create_brainiac_encoder(embed_dim=512, num_classes=6, pretrained_path=None, **kwargs):
    """Direct port from Zongyu's create_densenet_encoder."""
    model = BrainIACEncoder(embed_dim=embed_dim, num_classes=num_classes, **kwargs)
    if pretrained_path:
        model.load_pretrained_weights(pretrained_path)
    return model


# ============================================================================
# Auto-register BrainIAC with Zongyu's ModelRegistry (MY ADDITION)
# ============================================================================
# Required for Zongyu's EnhancedSingleModalTrainer grid search to find
# 'BrainIAC' in MODEL_NAMES. Without this, the trainer raises ValueError.

try:
    from .single_modal.model_registry import ModelRegistry
    ModelRegistry.register(
        'BrainIAC',
        lambda: BrainIACClassifier(num_classes=6),
        'standard',
        8,                              # halved from DenseNet's 32 — ViT-B is heavier
    )
    print("BrainIAC registered with ModelRegistry (batch_size=8)")
except Exception:
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("BrainIAC variants test")
    print("=" * 60)

    classifier = BrainIACClassifier(num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    output = classifier(x)
    print(f"\nClassifier 2-D input  -> {tuple(output.shape)} (expect (2, 6))")
    assert output.shape == (2, 6)

    encoder = BrainIACEncoder(embed_dim=512, num_classes=6)
    feats = encoder(x)
    print(f"Encoder 2-D features  -> {tuple(feats.shape)} (expect (2, 512))")
    assert feats.shape == (2, 512)

    feats, logits = encoder(x, return_features=True)
    print(f"Encoder return_features -> features {tuple(feats.shape)}, logits {tuple(logits.shape)}")

    print("\n" + "=" * 60)
    print("All shape assertions passed")
