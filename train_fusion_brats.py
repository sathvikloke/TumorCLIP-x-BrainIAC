"""BrainIAC + CLIP fusion training on BraTS 2024 — terminal entry point.

BraTS-specific sibling of train_fusion.py:
  - Reads a patient-level manifest CSV produced by scripts/inventory_brats.py
  - Uses src.data.brats_dataset.BraTSDataset (true 3-D NIfTI loading)
  - 3-class active-disease scheme (Quiescent / Enhancing / Necrotic enhancing)
  - Multilingual BRATS_DISEASE_ACTIVITY_PROMPTS for the text encoder
  - Inverse-frequency class weights applied to the CE terms to handle the
    48 / 29 / 23 class imbalance observed in the BraTS-GLI 2024 training cohort

The model architecture (BrainIAC encoder + CLIP text prototypes +
Tip-Adapter cache + SimpleFusionModel) is reused unchanged from
fusion_components.py and brainiac_variants.py.

Usage:
    python train_fusion_brats.py \\
        --manifest manifest_brats_train.csv \\
        --brainiac_weights weights/brainiac/BrainIAC.ckpt \\
        --modality t2f \\
        --epochs 20 \\
        --batch_size 4 \\
        --out_dir results/brats_fusion_seed41
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src/ is importable when run as `python train_fusion_brats.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.constants import (
    BRATS_CLASS_NAMES,
    BRATS_DISEASE_ACTIVITY_PROMPTS,
    BRATS_NUM_CLASSES,
)
from src.data.brats_dataset import make_brats_dataloaders
from src.models.fusion_components import (
    CLIPTextEncoder,
    OptimizedCLIPTipAdapter,
    SimpleFusionModel,
)
from src.models.brainiac_variants import BrainIACEncoder


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(manifest_csv: str, num_classes: int, device) -> torch.Tensor:
    """Inverse-frequency class weights from the training split."""
    df = pd.read_csv(manifest_csv)
    train_labels = df[(df["split"] == "train") & (df["label_idx"] >= 0)]["label_idx"]
    counts = Counter(int(x) for x in train_labels)
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        n_c = counts.get(c, 0)
        # inverse frequency, normalized so mean weight is ~1
        w = total / (num_classes * max(n_c, 1))
        weights.append(w)
    print(f"  class counts (train): {dict(counts)}")
    print(f"  class weights:        {['%.3f' % w for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Text-encoder prompt patching
# ---------------------------------------------------------------------------

def patch_text_encoder_for_brats(text_encoder: CLIPTextEncoder) -> None:
    """Override the text encoder's default prompts dict to BRATS_DISEASE_ACTIVITY_PROMPTS.

    The encoder's build_text_prototypes() falls back to PROFESSIONAL_MEDICAL_PROMPTS
    when called with prompts_dict=None. SimpleFusionModel's downstream
    OptimizedCLIPTipAdapter calls the encoder with no args, so we have to
    change the default at the instance level.
    """
    original_build = text_encoder.build_text_prototypes

    def _brats_build_text_prototypes(self, prompts_dict=None, device=None):
        return original_build(prompts_dict or BRATS_DISEASE_ACTIVITY_PROMPTS, device)

    text_encoder.build_text_prototypes = types.MethodType(
        _brats_build_text_prototypes, text_encoder
    )
    print("  Text encoder default prompts -> BRATS_DISEASE_ACTIVITY_PROMPTS")


# ---------------------------------------------------------------------------
# BraTS-aware Tip-Adapter cache builder
# ---------------------------------------------------------------------------

def build_brats_cache(image_encoder, train_loader, num_classes, device):
    """Build (cache_keys, cache_values) using BraTS class count, not Kaggle's."""
    image_encoder = image_encoder.to(device).eval()
    cache_keys = []
    cache_values = []
    with torch.no_grad():
        for volumes, labels in tqdm(train_loader, desc="Building BraTS cache"):
            volumes = volumes.to(device)
            labels = labels.to(device)
            feats = image_encoder.forward_features(volumes)
            feats = F.normalize(feats, dim=-1)
            one_hot = F.one_hot(labels, num_classes=num_classes).float()
            cache_keys.append(feats.cpu())
            cache_values.append(one_hot.cpu())
    cache_keys = torch.cat(cache_keys, dim=0)
    cache_values = torch.cat(cache_values, dim=0)
    print(f"  Cache built: keys {tuple(cache_keys.shape)}, "
          f"values {tuple(cache_values.shape)}")
    return cache_keys, cache_values


def attach_brats_clip_branch(
    model: SimpleFusionModel,
    train_loader,
    device,
    image_encoder,
    text_encoder,
    num_classes: int,
):
    """BraTS-aware replacement for SimpleFusionModel.setup_clip_branch.

    Functionally identical to the original but uses num_classes for the
    one-hot encoding in the cache (the original hard-codes Kaggle's 6).
    """
    print("Setting up BraTS CLIP branch")
    image_encoder = image_encoder.to(device).eval()
    text_encoder = text_encoder.to(device).eval()

    cache_keys, cache_values = build_brats_cache(
        image_encoder, train_loader, num_classes, device
    )

    model.clip_branch = OptimizedCLIPTipAdapter(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        cache_keys=cache_keys,
        cache_values=cache_values,
        alpha=model.clip_config["alpha"],
        t_knn=model.clip_config["t_knn"],
        lr_adapter=model.clip_config["lr_adapter"],
        device=device,
    ).to(device)

    model.image_encoder = image_encoder
    model.text_encoder = text_encoder
    model.device = device
    print(f"BraTS CLIP branch ready (alpha={model.clip_config['alpha']}, "
          f"t_knn={model.clip_config['t_knn']})")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    per_class_correct = Counter()
    per_class_total = Counter()
    for volumes, labels in loader:
        volumes = volumes.to(device)
        labels = labels.to(device)
        out = model(volumes, mode="eval")
        logits = out[0] if isinstance(out, tuple) else out
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            per_class_total[l] += 1
            if p == l:
                per_class_correct[l] += 1
    overall = correct / max(total, 1)
    per_class_acc = {
        c: per_class_correct[c] / max(per_class_total[c], 1)
        for c in sorted(per_class_total.keys())
    }
    return overall, per_class_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BraTS BrainIAC + CLIP fusion training")
    parser.add_argument("--manifest", required=True,
                        help="manifest_brats_train.csv from scripts/inventory_brats.py")
    parser.add_argument("--brainiac_weights", default=None,
                        help="Path to BrainIAC's .ckpt.")
    parser.add_argument("--modality", default="t2f",
                        choices=["t1n", "t1c", "t2w", "t2f"],
                        help="Single-modality input. Default t2f (FLAIR).")
    parser.add_argument("--multimodal", action="store_true",
                        help="Use all 4 modalities stacked along channel axis (4xDxHxW). "
                             "Note: BrainIAC was pretrained on single-channel inputs; "
                             "single-modality is the safer default for Phase 1.")
    parser.add_argument("--target_size", type=int, default=96,
                        help="Output cube side length. Default 96 to match BrainIAC.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="True 3-D 96^3 volumes are heavier than Kaggle slices "
                             "— start at 4 on H200, raise if memory allows.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shared_backbone", action="store_true", default=True,
                        help="Share the BrainIAC backbone between fusion branches "
                             "(default; halves GPU memory).")
    parser.add_argument("--no_shared_backbone", dest="shared_backbone",
                        action="store_false")
    parser.add_argument("--out_dir", default="results/brats_fusion")
    args = parser.parse_args()

    if args.brainiac_weights:
        os.environ["BRAINIAC_WEIGHTS_PATH"] = args.brainiac_weights

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir.resolve()}")

    # ----- Data ----------------------------------------------------------
    print("\n=== Building BraTS data loaders ===")
    train_loader, val_loader, test_loader = make_brats_dataloaders(
        manifest_csv=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        modality=args.modality,
        target_size=args.target_size,
        multimodal=args.multimodal,
        augment_train=True,
    )
    print(f"  train: {len(train_loader.dataset)} | "
          f"val: {len(val_loader.dataset)} | "
          f"test: {len(test_loader.dataset)}")
    print(f"  modality: {args.modality} | multimodal: {args.multimodal} | "
          f"target_size: {args.target_size}^3")

    # ----- Class weights -------------------------------------------------
    print("\n=== Computing class weights from training set ===")
    class_weights = compute_class_weights(args.manifest, BRATS_NUM_CLASSES, device)

    # ----- Encoders ------------------------------------------------------
    print("\n=== Building encoders ===")
    image_encoder = BrainIACEncoder(embed_dim=512, dropout=0.1)
    text_encoder = CLIPTextEncoder(class_names=BRATS_CLASS_NAMES)
    patch_text_encoder_for_brats(text_encoder)

    if args.brainiac_weights and os.path.exists(args.brainiac_weights):
        try:
            image_encoder.load_pretrained_weights(args.brainiac_weights)
            print(f"  Loaded BrainIAC weights into image_encoder.")
        except Exception as e:
            print(f"  WARN: failed to load image_encoder weights: {e}")

    # ----- Fusion model --------------------------------------------------
    print("\n=== Building fusion model ===")
    BEST_BRAINIAC_CONFIG = {
        "backbone_lr": 5e-5, "head_lr": 1e-3,
        "focal_gamma": 2.0, "label_smoothing": 0.05,
    }
    BEST_CLIP_CONFIG = {"alpha": 0.5, "t_knn": 0.07, "lr_adapter": 3e-4}
    model = SimpleFusionModel(
        brainiac_config=BEST_BRAINIAC_CONFIG,
        clip_config=BEST_CLIP_CONFIG,
        num_classes=BRATS_NUM_CLASSES,
    ).to(device)

    if args.brainiac_weights and os.path.exists(args.brainiac_weights):
        try:
            model.load_brainiac_weights(args.brainiac_weights)
            print(f"  Loaded BrainIAC weights into fusion.brainiac_branch.")
        except Exception as e:
            print(f"  WARN: failed to load brainiac_branch weights: {e}")

    if args.shared_backbone:
        image_encoder = image_encoder.to(device)
        model.share_backbone_with(image_encoder)

    # ----- CLIP branch (BraTS-aware) ------------------------------------
    print("\n=== Attaching CLIP branch ===")
    attach_brats_clip_branch(
        model, train_loader, device, image_encoder, text_encoder,
        num_classes=BRATS_NUM_CLASSES,
    )

    # ----- Optimizer + scheduler ----------------------------------------
    optimizer = torch.optim.AdamW(model.get_optimizer_params(), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # ----- Training loop -------------------------------------------------
    print("\n=== Training ===")
    history = {
        "train_loss": [],
        "val_acc": [], "val_per_class_acc": [],
        "test_acc": [], "test_per_class_acc": [],
        "fusion_weight_sigma": [],
    }
    best_val_acc = -1.0

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, n = 0.0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
            for volumes, labels in pbar:
                volumes = volumes.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                out = model(volumes, mode="train")
                if isinstance(out, tuple) and len(out) == 3:
                    fused_logits, brainiac_logits, clip_logits = out
                    fusion_loss = F.cross_entropy(
                        fused_logits, labels, weight=class_weights,
                    )
                    brainiac_loss = model.brainiac_branch.compute_loss(
                        brainiac_logits, labels, loss_type="focal",
                    )
                    clip_loss = F.cross_entropy(
                        clip_logits, labels, weight=class_weights,
                    )
                    loss = 0.5 * fusion_loss + 0.3 * brainiac_loss + 0.2 * clip_loss
                else:
                    loss = F.cross_entropy(out, labels, weight=class_weights)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * volumes.size(0)
                n += volumes.size(0)
                pbar.set_postfix(loss=f"{running_loss / max(n, 1):.4f}")

            scheduler.step()
            train_loss = running_loss / max(n, 1)

            val_acc, val_per_class = evaluate(model, val_loader, device)
            test_acc, test_per_class = evaluate(model, test_loader, device)
            fw = torch.sigmoid(model.fusion_weight).item()

            print(
                f"  [Epoch {epoch:02d}] "
                f"train_loss={train_loss:.4f}  "
                f"val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  "
                f"sigma(fw)={fw:.3f}"
            )
            print(f"     val per-class: "
                  f"{[f'{c}:{v:.3f}' for c, v in sorted(val_per_class.items())]}")

            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            history["val_per_class_acc"].append(val_per_class)
            history["test_acc"].append(test_acc)
            history["test_per_class_acc"].append(test_per_class)
            history["fusion_weight_sigma"].append(fw)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                        "val_per_class_acc": val_per_class,
                        "test_per_class_acc": test_per_class,
                        "args": vars(args),
                        "class_weights": class_weights.cpu().tolist(),
                    },
                    out_dir / "best.pt",
                )
                print(f"     -> new best val_acc; saved {out_dir / 'best.pt'}")
    finally:
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nHistory written to {out_dir / 'history.json'}")
        print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
