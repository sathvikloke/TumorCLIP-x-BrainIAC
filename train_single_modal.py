"""BrainIAC single-modal (stage 1) training — terminal entry point.

Fine-tunes BrainIAC's classifier head on the 6-class tumor task. The
output checkpoint can then be fed into train_fusion.py via
--brainiac_weights.

Usage:
    python train_single_modal.py \
        --data_root data \
        --brainiac_weights weights/brainiac/BrainIAC.ckpt \
        --epochs 40 \
        --batch_size 8 \
        --lr 1e-4 \
        --optimizer adam \
        --out_dir results/best_models
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.constants import CLASS_NAMES, NUM_CLASSES
from src.models.brainiac_variants import BrainIACClassifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_data_loaders(data_root: str, batch_size: int, seed: int, num_workers: int = 4):
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    for d in (train_dir, test_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing dataset directory: {d}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train = datasets.ImageFolder(train_dir, transform=train_transform)
    test_ds = datasets.ImageFolder(test_dir, transform=val_test_transform)

    n_train = int(0.8 * len(full_train))
    n_val = len(full_train) - n_train
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=gen)
    val_ds.dataset.transform = val_test_transform  # type: ignore[attr-defined]

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, generator=gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="BrainIAC single-modal stage-1 training")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--brainiac_weights", default=None,
                        help="Path to BrainIAC's .ckpt. Overrides BRAINIAC_WEIGHTS_PATH env var.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Head learning rate. Backbone uses lr / 20 by default.")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience in epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", default="results/best_models")
    args = parser.parse_args()

    if args.brainiac_weights:
        os.environ["BRAINIAC_WEIGHTS_PATH"] = args.brainiac_weights

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Building data loaders ===")
    train_loader, val_loader, test_loader = build_data_loaders(
        args.data_root, args.batch_size, args.seed, args.num_workers,
    )
    print(f"  train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, "
          f"test: {len(test_loader.dataset)}")

    print("\n=== Building BrainIAC classifier ===")
    model = BrainIACClassifier(
        num_classes=NUM_CLASSES,
        backbone_lr=args.lr / 20.0,
        head_lr=args.lr,
        focal_gamma=2.0,
        label_smoothing=0.05,
        weights_path=args.brainiac_weights,
    ).to(device)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.get_optimizer_params(), weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.get_optimizer_params(), momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\n=== Training ===")
    history = {"train_loss": [], "val_acc": [], "test_acc": []}
    best_val_acc = -1.0
    epochs_no_improve = 0

    out_name = f"BrainIAC_{args.optimizer.capitalize()}_lr{args.lr}_best.pth"

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, n = 0.0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
            for images, labels in pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = model.compute_loss(logits, labels, loss_type="focal")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                n += images.size(0)
                pbar.set_postfix(loss=f"{running_loss / max(n, 1):.4f}")

            scheduler.step()
            train_loss = running_loss / max(n, 1)
            val_acc = evaluate(model, val_loader, device)
            test_acc = evaluate(model, test_loader, device)
            print(f"  [Epoch {epoch}] train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
                  f"test_acc={test_acc:.4f}")

            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            history["test_acc"].append(test_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": epoch,
                     "val_acc": val_acc, "args": vars(args)},
                    out_dir / out_name,
                )
                print(f"  -> new best val_acc; saved {out_dir / out_name}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement in {args.patience} epochs).")
                    break
    finally:
        with open(out_dir / "single_modal_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nHistory written to {out_dir / 'single_modal_history.json'}")


if __name__ == "__main__":
    main()
