"""BrainIAC + CLIP fusion training — terminal entry point.

Replaces BrainIAC_CLIP_Fusion_Model_Training.ipynb with a CLI script you
can run from VS Code's terminal:

    python train_fusion.py \
        --data_root data \
        --brainiac_weights weights/brainiac/BrainIAC.ckpt \
        --epochs 15 \
        --batch_size 8 \
        --out_dir results/fusion

The script logic is the same as the notebook's: build BrainIAC encoder +
CLIP text encoder, build Tip-Adapter cache from training data, train the
SimpleFusionModel with multi-task loss, save the best checkpoint.
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

# Ensure src/ is on path when run as `python train_fusion.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.constants import CLASS_NAMES, NUM_CLASSES
from src.models.fusion_components import SimpleFusionModel, create_clip_brainiac_model


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
# Data
# ---------------------------------------------------------------------------

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

    expected = sorted(CLASS_NAMES)
    if sorted(full_train.classes) != expected:
        print(
            f"WARNING: ImageFolder classes {full_train.classes} do not match "
            f"CLASS_NAMES {CLASS_NAMES}. Re-check directory names."
        )

    n_train = int(0.8 * len(full_train))
    n_val = len(full_train) - n_train
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=gen)
    val_ds.dataset.transform = val_test_transform  # type: ignore[attr-defined]

    def _worker_init(worker_id: int) -> None:
        ws = torch.initial_seed() % 2 ** 32
        np.random.seed(ws)
        random.seed(ws)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, worker_init_fn=_worker_init, generator=gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
        pin_memory=pin, worker_init_fn=_worker_init,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
        pin_memory=pin, worker_init_fn=_worker_init,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        out = model(images, mode="eval")
        logits = out[0] if isinstance(out, tuple) else out
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BrainIAC + CLIP fusion training")
    parser.add_argument("--data_root", default="data",
                        help="Directory with train/ and test/ subdirs (ImageFolder layout)")
    parser.add_argument("--brainiac_weights", default=None,
                        help="Path to BrainIAC's .ckpt. Overrides BRAINIAC_WEIGHTS_PATH env var.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="ViT-B with 96^3 input is heavier than DenseNet — start at 8.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shared_backbone", action="store_true", default=True,
                        help="Share the BrainIAC backbone between fusion branches (default; "
                             "halves GPU memory). Pass --no_shared_backbone to disable.")
    parser.add_argument("--no_shared_backbone", dest="shared_backbone", action="store_false")
    parser.add_argument("--out_dir", default="results/fusion")
    args = parser.parse_args()

    if args.brainiac_weights:
        os.environ["BRAINIAC_WEIGHTS_PATH"] = args.brainiac_weights

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("\n=== Building data loaders ===")
    train_loader, val_loader, test_loader = build_data_loaders(
        args.data_root, args.batch_size, args.seed, args.num_workers,
    )
    print(f"  train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, "
          f"test: {len(test_loader.dataset)}")

    # Encoders (CLIP + BrainIAC for the CLIP branch)
    print("\n=== Building encoders ===")
    image_encoder, text_encoder = create_clip_brainiac_model(embed_dim=512, dropout=0.1)
    if args.brainiac_weights and os.path.exists(args.brainiac_weights):
        image_encoder.load_pretrained_weights(args.brainiac_weights)

    # Fusion model
    print("\n=== Building fusion model ===")
    BEST_BRAINIAC_CONFIG = {
        "backbone_lr": 5e-5, "head_lr": 1e-3,
        "focal_gamma": 2.0, "label_smoothing": 0.05,
    }
    BEST_CLIP_CONFIG = {"alpha": 0.5, "t_knn": 0.07, "lr_adapter": 3e-4}
    model = SimpleFusionModel(
        brainiac_config=BEST_BRAINIAC_CONFIG,
        clip_config=BEST_CLIP_CONFIG,
        num_classes=NUM_CLASSES,
    ).to(device)

    # Load BrainIAC weights into the brainiac_branch too
    if args.brainiac_weights and os.path.exists(args.brainiac_weights):
        model.load_brainiac_weights(args.brainiac_weights)

    # Optional: share the backbone (halves GPU memory)
    if args.shared_backbone:
        image_encoder = image_encoder.to(device)
        model.share_backbone_with(image_encoder)

    # Set up CLIP branch (builds the cache)
    model.setup_clip_branch(train_loader, device, image_encoder, text_encoder)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.get_optimizer_params(), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n=== Training ===")
    history = {"train_loss": [], "val_acc": [], "test_acc": []}
    best_val_acc = -1.0

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, n = 0.0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
            for images, labels in pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                out = model(images, mode="train")
                if isinstance(out, tuple) and len(out) == 3:
                    fused_logits, brainiac_logits, clip_logits = out
                    fusion_loss = F.cross_entropy(fused_logits, labels)
                    brainiac_loss = model.brainiac_branch.compute_loss(
                        brainiac_logits, labels, loss_type="focal"
                    )
                    clip_loss = F.cross_entropy(clip_logits, labels)
                    loss = 0.5 * fusion_loss + 0.3 * brainiac_loss + 0.2 * clip_loss
                else:
                    loss = F.cross_entropy(out, labels)

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
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch,
                     "val_acc": val_acc, "args": vars(args)},
                    out_dir / "best.pt",
                )
                print(f"  -> new best val_acc; saved {out_dir / 'best.pt'}")
    finally:
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nHistory written to {out_dir / 'history.json'}")


if __name__ == "__main__":
    main()
