"""Inventory BraTS 2024 patients and generate a labelled, split manifest.

Walks an extracted BraTS-GLI 2024 directory, verifies modality completeness
per patient, derives 4-class active-disease labels from the segmentation
masks, and produces a manifest CSV with deterministic patient-level
train/val/test splits.

BraTS 2024 GLI seg labels:
    0 = background
    1 = NCR (necrotic / non-enhancing tumor core)
    2 = ED  (peritumoral edema / infiltrative edema)
    3 = ET  (enhancing tumor)
    4 = RC  (resection cavity, post-treatment cases only)

Cohort observation (June 2026): the BraTS-GLI 2024 training cohort is
~85% post-treatment (label 4 present) and ~15% pre-treatment. Class scheme
is therefore designed around residual *active disease* (NCR and ET
presence), not treatment status. RC is metadata-tracked but not used to
distinguish classes.

Derived classes (presence-based):
    0 - Quiescent / clean cavity (no NCR, low ET) - treatment effect or
        minimal residual disease
    1 - Enhancing without necrosis (ET present, NCR absent)
    2 - Necrotic non-enhancing (NCR present, ET absent)
    3 - Necrotic enhancing (NCR + ET both present) - most aggressive pattern

Usage:
    python scripts/inventory_brats.py \
        --brats_root ~/brats2024/extracted/training \
        --out_manifest manifest_brats_train.csv \
        --val_frac 0.15 --test_frac 0.15 --seed 42 --require_seg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# ==================== Constants ====================

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

# BraTS 2024 GLI segmentation labels
SEG_LABEL_NECROTIC = 1   # NCR
SEG_LABEL_EDEMA = 2      # ED
SEG_LABEL_ENHANCING = 3  # ET
SEG_LABEL_CAVITY = 4     # RC (post-treatment only)

CLASS_NAMES = [
    "Quiescent (minimal active disease)",
    "Enhancing without necrosis",
    "Necrotic enhancing (ring-enhancing)",
]
NUM_CLASSES = len(CLASS_NAMES)

# Presence thresholds: a label is "present" if its fraction of the total
# lesion (NCR+ED+ET+RC) exceeds this. Tuned for the post-treatment-heavy
# 2024 cohort where small NCR/ET volumes are common in residual disease.
NCR_PRESENCE_THRESH = 0.01   # NCR is "present" at >= 1% of lesion
ET_PRESENCE_THRESH = 0.03    # ET is "present" at >= 3% of lesion


# ==================== Label derivation ====================

def derive_label_from_seg(seg_path: Path) -> tuple[int, dict]:
    """Derive a 4-class active-disease label from the seg mask.

    Returns (label_idx, info_dict) where info_dict tracks voxel counts and
    fractional composition for diagnostic logging.
    """
    seg = nib.load(str(seg_path)).get_fdata().astype(np.int32)

    n_necrotic = int((seg == SEG_LABEL_NECROTIC).sum())
    n_edema = int((seg == SEG_LABEL_EDEMA).sum())
    n_enhancing = int((seg == SEG_LABEL_ENHANCING).sum())
    n_cavity = int((seg == SEG_LABEL_CAVITY).sum())
    n_total = n_necrotic + n_edema + n_enhancing + n_cavity

    info = {
        "vox_necrotic": n_necrotic,
        "vox_edema": n_edema,
        "vox_enhancing": n_enhancing,
        "vox_cavity": n_cavity,
        "vox_total_lesion": n_total,
        "has_cavity": bool(n_cavity > 100),
    }

    if n_total == 0:
        info["frac_necrotic"] = 0.0
        info["frac_edema"] = 0.0
        info["frac_enhancing"] = 0.0
        info["frac_cavity"] = 0.0
        return -1, info  # no lesion at all - exclude

    ncr_frac = n_necrotic / n_total
    ed_frac = n_edema / n_total
    et_frac = n_enhancing / n_total
    rc_frac = n_cavity / n_total
    info["frac_necrotic"] = ncr_frac
    info["frac_edema"] = ed_frac
    info["frac_enhancing"] = et_frac
    info["frac_cavity"] = rc_frac

    has_ncr = ncr_frac > NCR_PRESENCE_THRESH
    has_et = et_frac > ET_PRESENCE_THRESH

    # 3-class scheme: NCR-without-ET (rare ~1%) is excluded as label -1
    if has_ncr and has_et:
        return 2, info  # Necrotic enhancing (most aggressive)
    if has_ncr and not has_et:
        return -1, info # Exclude: too few patients to train (~1% of cohort)
    if not has_ncr and has_et:
        return 1, info  # Enhancing without necrosis
    return 0, info      # Quiescent / minimal active disease


# ==================== Filesystem walk ====================

def find_patient_dirs(brats_root: Path) -> list[Path]:
    """Find every BraTS-GLI-* patient directory under brats_root.

    Robust to nested extraction (sometimes archives wrap content in an extra
    folder). Verifies at least one expected modality file exists inside.
    """
    candidates = set()
    for child in brats_root.rglob("BraTS-GLI-*"):
        if not child.is_dir():
            continue
        has_modality = any(child.glob(f"*-{m}.nii.gz") for m in MODALITIES)
        if has_modality:
            candidates.add(child)
    return sorted(candidates)


def verify_patient(patient_dir: Path) -> dict:
    """Verify modality completeness for one patient."""
    patient_id = patient_dir.name
    info = {
        "patient_id": patient_id,
        "patient_dir": str(patient_dir),
        "complete_modalities": True,
    }

    for m in MODALITIES:
        f = patient_dir / f"{patient_id}-{m}.nii.gz"
        info[f"has_{m}"] = f.exists()
        if not f.exists():
            info["complete_modalities"] = False

    seg_f = patient_dir / f"{patient_id}-seg.nii.gz"
    info["has_seg"] = seg_f.exists()
    info["seg_path"] = str(seg_f) if seg_f.exists() else ""

    return info


# ==================== Splits ====================

def make_splits(
    patient_ids: list[str], val_frac: float, test_frac: float, seed: int
) -> dict[str, str]:
    """Deterministic patient-level train/val/test split.

    Uses numpy's default_rng with a fixed seed for reproducibility.
    """
    assert 0 <= val_frac < 1 and 0 <= test_frac < 1
    assert val_frac + test_frac < 1

    rng = np.random.default_rng(seed)
    shuffled = list(patient_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test

    split_map: dict[str, str] = {}
    for pid in shuffled[:n_train]:
        split_map[pid] = "train"
    for pid in shuffled[n_train:n_train + n_val]:
        split_map[pid] = "val"
    for pid in shuffled[n_train + n_val:]:
        split_map[pid] = "test"

    return split_map


# ==================== Diagnostics ====================

def print_summary(df: pd.DataFrame) -> None:
    print()
    print("===== Summary =====")
    print(f"Total patients in manifest: {len(df)}")

    if "label_idx" in df.columns and (df["label_idx"] >= 0).any():
        valid = df[df["label_idx"] >= 0]
        print("\nLabel distribution overall:")
        for cls_idx in range(NUM_CLASSES):
            n = int((valid["label_idx"] == cls_idx).sum())
            pct = 100 * n / len(valid) if len(valid) else 0
            print(f"  {cls_idx} {CLASS_NAMES[cls_idx]:<42s} {n:6d}  ({pct:5.1f}%)")

    if "has_cavity" in df.columns:
        n_post = int(df["has_cavity"].sum())
        n_pre = len(df) - n_post
        pct_post = 100 * n_post / len(df) if len(df) else 0
        print(f"\nCohort treatment status (informational):")
        print(f"  Post-treatment (has RC): {n_post} ({pct_post:.1f}%)")
        print(f"  Pre-treatment (no RC):   {n_pre} ({100 - pct_post:.1f}%)")

    if "split" in df.columns:
        print("\nSplit sizes:")
        for split_name, count in df["split"].value_counts().items():
            print(f"  {split_name:<12s} {count}")

        if "label_idx" in df.columns and (df["label_idx"] >= 0).any():
            print("\nLabel x split crosstab (counts):")
            ct = pd.crosstab(df["label_idx"], df["split"]).reindex(
                range(NUM_CLASSES), fill_value=0
            )
            print(ct.to_string())


# ==================== Main ====================

def main():
    p = argparse.ArgumentParser(
        description="Inventory BraTS 2024 patients and emit a labelled manifest CSV.",
    )
    p.add_argument(
        "--brats_root",
        required=True,
        help="Directory containing BraTS-GLI-* patient folders (extracted).",
    )
    p.add_argument(
        "--out_manifest",
        default="manifest_brats.csv",
        help="Output CSV path.",
    )
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--require_seg",
        action="store_true",
        help="Skip patients with no segmentation mask "
        "(necessary for label derivation; validation split has no masks).",
    )
    p.add_argument(
        "--require_modalities",
        action="store_true",
        default=True,
        help="Skip patients missing any of the 4 modalities.",
    )
    args = p.parse_args()

    brats_root = Path(args.brats_root).expanduser().resolve()
    if not brats_root.exists():
        print(f"ERROR: brats_root does not exist: {brats_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Walking {brats_root} ...")
    patient_dirs = find_patient_dirs(brats_root)
    print(f"Found {len(patient_dirs)} candidate patient directories.")

    rows = []
    skipped_incomplete = 0
    skipped_no_seg = 0
    skipped_label_fail = 0

    for pdir in patient_dirs:
        info = verify_patient(pdir)

        if args.require_modalities and not info["complete_modalities"]:
            skipped_incomplete += 1
            continue
        if args.require_seg and not info["has_seg"]:
            skipped_no_seg += 1
            continue

        if info["has_seg"]:
            try:
                label, label_info = derive_label_from_seg(Path(info["seg_path"]))
            except Exception as e:
                print(f"  WARN: seg load failed for {info['patient_id']}: {e}")
                skipped_label_fail += 1
                continue
            info["label_idx"] = label
            info["label_name"] = CLASS_NAMES[label] if 0 <= label < NUM_CLASSES else ""
            info.update(label_info)
        else:
            info["label_idx"] = -1
            info["label_name"] = ""

        rows.append(info)

    print(f"  Kept:                 {len(rows)}")
    print(f"  Skipped (no modalities): {skipped_incomplete}")
    print(f"  Skipped (no seg mask):   {skipped_no_seg}")
    print(f"  Skipped (label failed):  {skipped_label_fail}")

    if not rows:
        print("\nNo valid patients found - exiting without writing manifest.", file=sys.stderr)
        sys.exit(2)

    pids_for_splits = [r["patient_id"] for r in rows if r["label_idx"] >= 0]
    split_map = make_splits(pids_for_splits, args.val_frac, args.test_frac, args.seed)
    for r in rows:
        r["split"] = split_map.get(r["patient_id"], "unlabelled")

    df = pd.DataFrame(rows)

    preferred = [
        "patient_id", "patient_dir", "split", "label_idx", "label_name",
        "complete_modalities", "has_seg", "has_cavity",
        "has_t1n", "has_t1c", "has_t2w", "has_t2f", "seg_path",
        "vox_total_lesion", "vox_necrotic", "vox_edema", "vox_enhancing", "vox_cavity",
        "frac_necrotic", "frac_edema", "frac_enhancing", "frac_cavity",
    ]
    cols = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    df = df[cols]

    out_path = Path(args.out_manifest).expanduser().resolve()
    df.to_csv(out_path, index=False)
    print(f"\nWrote manifest: {out_path}")

    print_summary(df)


if __name__ == "__main__":
    main()
