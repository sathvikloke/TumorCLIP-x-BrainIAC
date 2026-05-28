"""
Dataset loader for the processed Kaggle Brain Tumor MRI dataset.

Assumes `scripts/download_data.sh` + `scripts/consolidate_classes.py` have
already produced the directory layout:

    data/processed/
        train/<class_name>/<img>.jpg
        test/<class_name>/<img>.jpg

where <class_name> is one of the six superclasses defined in prototypes.py.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .prototypes import CLASS_NAMES


# ImageNet normalization, as in the original TumorCLIP paper.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Construct image transforms.

    The paper uses simple augmentations (resize + normalize). We add a light
    flip during training to match the "standardized augmentations" referenced
    in the paper.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class KaggleTumorDataset(Dataset):
    """6-class brain tumor dataset (post 17->6 consolidation)."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 224,
    ):
        super().__init__()
        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(
                f"Expected {self.root}. Run scripts/download_data.sh first."
            )

        self.transform = build_transforms(image_size, train=(split == "train"))
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.samples: list[tuple[Path, int]] = []

        for class_name in CLASS_NAMES:
            class_dir = self.root / class_name
            if not class_dir.exists():
                # Empty class — warn but allow (e.g. if a class has no test samples)
                continue
            for img_path in sorted(class_dir.iterdir()):
                if (
                    img_path.is_file()
                    and not img_path.name.startswith(".")
                    and img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                ):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. Check the data layout."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for handling class imbalance."""
        counts = torch.zeros(len(CLASS_NAMES))
        for _, label in self.samples:
            counts[label] += 1
        # Avoid div-by-zero on empty classes
        counts = torch.clamp(counts, min=1.0)
        weights = 1.0 / counts
        return weights / weights.sum() * len(CLASS_NAMES)
