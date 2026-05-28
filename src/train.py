"""
Training entry point.

Usage:
    python -m src.train --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import KaggleTumorDataset
from .evaluate import evaluate
from .model import TimmBackbone, TumorCLIP
from .prototypes import build_prototype_bank, CLASS_NAMES


def set_seed(seed: int) -> None:
    """Seed RNGs + force deterministic cuDNN.

    Determinism matters for this project specifically: every result is part of
    a head-to-head comparison across backbone / head / fine-tuning variants,
    so re-running the same config must produce the same number.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> TumorCLIP:
    # Frozen text encoder produces prototypes once
    bank = build_prototype_bank(
        model_name=cfg["text_encoder"]["name"],
        pretrained=cfg["text_encoder"]["pretrained"],
        device=device,
    )

    backbone = TimmBackbone(
        name=cfg["backbone"]["name"],
        pretrained=cfg["backbone"]["pretrained"],
        freeze=cfg["backbone"].get("freeze", False),
    )

    model = TumorCLIP(
        backbone=backbone,
        text_prototypes=bank.embeddings,
        n_classes=len(CLASS_NAMES),
        alpha=cfg["head"]["alpha"],
        projection_hidden=cfg["head"]["projection_hidden"],
        learnable_alpha=cfg["head"].get("learnable_alpha", False),
    ).to(device)

    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0
    pbar = tqdm(loader, desc=f"Train epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = model(images)
        loss = criterion(out["logits"], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        pbar.set_postfix(loss=f"{running_loss / n_samples:.4f}")

    return running_loss / max(n_samples, 1)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_ds = KaggleTumorDataset(cfg["data"]["root"], "train", cfg["data"]["image_size"])
    test_ds = KaggleTumorDataset(cfg["data"]["root"], "test", cfg["data"]["image_size"])
    pin = device.type == "cuda"  # pin_memory is only meaningful on CUDA
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=pin,
    )

    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    # Model
    model = build_model(cfg, device)

    # Optimizer + scheduler + loss
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"].get("label_smoothing", 0.0))

    # Optional W&B — wrap the whole training body in try/finally so the run
    # is finalized cleanly even if training crashes.
    use_wandb = cfg["logging"].get("use_wandb", False)
    wandb = None
    if use_wandb:
        import wandb  # local import keeps wandb optional
        wandb.init(project=cfg["logging"]["wandb_project"], name=cfg["experiment_name"], config=cfg)

    # Training loop with early stopping on test macro-F1
    best_f1 = -1.0
    epochs_no_improve = 0
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
            scheduler.step()

            result = evaluate(model, test_loader, device)
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} "
                f"test_acc={result.accuracy:.4f} test_macroF1={result.macro_f1:.4f}"
            )

            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "test/accuracy": result.accuracy,
                    "test/macro_f1": result.macro_f1,
                    **{f"test/recall_{k}": v for k, v in result.per_class_recall.items()},
                })

            if result.macro_f1 > best_f1:
                best_f1 = result.macro_f1
                epochs_no_improve = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "macro_f1": best_f1,
                        "config": cfg,
                    },
                    ckpt_dir / f"{cfg['experiment_name']}_best.pt",
                )
                print(f"  -> new best macroF1, checkpoint saved.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg["training"]["early_stopping_patience"]:
                    print(f"Early stopping after {epoch} epochs (best macroF1: {best_f1:.4f})")
                    break

        # Final report from the best checkpoint. weights_only=False because the
        # checkpoint dict embeds the config (non-tensor Python objects).
        best_ckpt = torch.load(
            ckpt_dir / f"{cfg['experiment_name']}_best.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_ckpt["model"])
        final = evaluate(model, test_loader, device)
        print("\n=== Final test results (best checkpoint) ===")
        print(f"Accuracy: {final.accuracy:.4f}")
        print(f"Macro-F1: {final.macro_f1:.4f}")
        print(f"Per-class recall: {final.per_class_recall}")
        print(final.report)
    finally:
        if use_wandb and wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
