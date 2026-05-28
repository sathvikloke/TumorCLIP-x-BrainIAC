"""
Map the Kaggle 17-class brain tumor dataset to the 6 superclasses used in the
TumorCLIP paper.

The exact 17->6 mapping isn't given in the paper; the mapping below is the
clinically reasonable consolidation that matches the paper's six categories:
Glioma, Meningioma, Normal, Neurocytoma, Other Lesions, Schwannoma.

Edit MAPPING below to adjust. Run after `download_data.sh`.

Usage:
    python scripts/consolidate_classes.py \\
        --raw_dir data/raw \\
        --out_dir data/processed \\
        --test_frac 0.35 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

# 17-class names from the Kaggle dataset (fernando2rad/brain-tumor-mri-images-17-classes)
# may differ slightly in casing or wording — adjust if needed after inspecting raw_dir.
MAPPING: dict[str, str] = {
    # Gliomas
    "Astrocitoma": "Glioma",
    "Glioblastoma": "Glioma",
    "Oligodendroglioma": "Glioma",
    "Ependimoma": "Glioma",
    # Meningiomas
    "Meningioma": "Meningioma",
    # Normal
    "_NORMAL": "Normal",
    "NORMAL": "Normal",
    # Neurocytoma
    "Neurocitoma": "Neurocytoma",
    # Schwannoma
    "Schwannoma": "Schwannoma",
    # Catch-all "Other lesions" — anything not in the above maps here.
    # Sub-categories from the Kaggle dataset (Carcinoma, Ganglioglioma, etc.)
    "Carcinoma": "Other Lesions",
    "Ganglioglioma": "Other Lesions",
    "Granuloma": "Other Lesions",
    "Meduloblastoma": "Other Lesions",
    "Papiloma": "Other Lesions",
    "Tuberculoma": "Other Lesions",
    "Germinoma": "Other Lesions",
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_superclass(folder_name: str, strict: bool = False) -> str | None:
    """Map a raw class folder name to its superclass.

    The Kaggle dataset uses names like "Astrocitoma T1", "Astrocitoma T1C+",
    "Astrocitoma T2" — one folder per (subclass, contrast) pair. We strip the
    contrast suffix and look up in MAPPING. If not strict, unmapped folders
    fall through to 'Other Lesions'.
    """
    name = folder_name.strip()

    # Try exact match first
    if name in MAPPING:
        return MAPPING[name]

    # Try stripping common contrast suffixes
    for suffix in (" T1C+", " T1c+", " T1CE", " T1C", " T1", " T2", " FLAIR", " ADC", " DWI"):
        if name.endswith(suffix):
            base = name[: -len(suffix)].strip()
            if base in MAPPING:
                return MAPPING[base]

    # Try prefix substring match (case-insensitive) against known keys
    lower = name.lower()
    for key, super_cls in MAPPING.items():
        if lower.startswith(key.lower()):
            return super_cls

    if strict:
        return None
    return "Other Lesions"


def stratified_split(
    files_per_class: dict[str, list[Path]],
    test_frac: float,
    seed: int,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    rng = random.Random(seed)
    train: dict[str, list[Path]] = {}
    test: dict[str, list[Path]] = {}
    for cls, files in files_per_class.items():
        shuffled = files[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_frac))
        test[cls] = shuffled[:n_test]
        train[cls] = shuffled[n_test:]
    return train, test


def main(raw_dir: str, out_dir: str, test_frac: float, seed: int, strict: bool = False) -> None:
    raw = Path(raw_dir)
    out = Path(out_dir)
    if not raw.exists():
        raise FileNotFoundError(f"Raw data dir not found: {raw}")

    # If raw_dir contains exactly one subdir and no image files, descend into it
    # (the Kaggle zip typically extracts to data/raw/<dataset_name>/).
    children = [c for c in raw.iterdir() if c.is_dir()]
    has_imgs_here = any(p.suffix.lower() in VALID_EXTS for c in children for p in c.iterdir())
    if len(children) == 1 and not has_imgs_here:
        raw = children[0]
        print(f"[info] Descending into single subdir: {raw}")

    # Collect images per superclass
    files_per_super: dict[str, list[Path]] = {}
    fallback_assignments: list[tuple[str, str]] = []  # (raw_name, fallback_super)
    unmapped: list[str] = []
    for class_dir in sorted(raw.iterdir()):
        if not class_dir.is_dir():
            continue
        # If strict resolution would have failed, we know we used the catch-all.
        strict_super = resolve_superclass(class_dir.name, strict=True)
        super_cls = strict_super if strict_super is not None else resolve_superclass(
            class_dir.name, strict=strict
        )
        if super_cls is None:
            unmapped.append(class_dir.name)
            continue
        if strict_super is None and super_cls == "Other Lesions":
            fallback_assignments.append((class_dir.name, super_cls))
        bucket = files_per_super.setdefault(super_cls, [])
        for p in class_dir.iterdir():
            if p.suffix.lower() in VALID_EXTS:
                bucket.append(p)

    if fallback_assignments:
        print(f"[info] {len(fallback_assignments)} folders fell through to 'Other Lesions':")
        for name, _ in fallback_assignments[:10]:
            print(f"         {name}")
        if len(fallback_assignments) > 10:
            print(f"         (...{len(fallback_assignments) - 10} more)")
    if unmapped:
        print(f"[warn] Unmapped class folders (skipped, --strict mode): {unmapped}")

    print("Counts per superclass:")
    for k, v in sorted(files_per_super.items()):
        print(f"  {k}: {len(v)}")

    train, test = stratified_split(files_per_super, test_frac, seed)

    # Copy into out_dir/{train,test}/<superclass>/
    for split_name, split in (("train", train), ("test", test)):
        for cls, files in split.items():
            target = out / split_name / cls
            target.mkdir(parents=True, exist_ok=True)
            for src in files:
                dst = target / f"{src.parent.name}__{src.name}"
                shutil.copyfile(src, dst)
        n = sum(len(v) for v in split.values())
        print(f"Wrote {n} images to {out / split_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--test_frac", type=float, default=0.35,
                        help="Fraction of each class to use for test (paper uses ~35%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true",
                        help="Skip unmapped classes instead of routing them to 'Other Lesions'.")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.test_frac, args.seed, args.strict)
