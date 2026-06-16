"""BraTS concept-intervention falsification sweep.

Loads a trained BraTS checkpoint (from train_fusion_brats.py), rebuilds
the model architecture, swaps in edited text prototypes per class, and
measures the prediction shift for each of the 3 BraTS classes.

For each class we run:
  - one meaningful clinical token swap
  - one matched semantically null control swap

A faithful interpretability mechanism means the meaningful edit should
produce a markedly larger probability drop / flip rate than the control.

Usage:
    python scripts/concept_intervention_brats.py \\
        --checkpoint results/brats_fusion_seed41/best.pt \\
        --manifest manifest_brats_train.csv \\
        --modality t2f \\
        --batch_size 8 \\
        --out_json results/brats_fusion_seed41/intervention_seed41.json

For the sweep to be statistically meaningful, repeat across 3+ seeds and
aggregate the per-class probability drops with mean ± std.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow `python scripts/concept_intervention_brats.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.constants import BRATS_CLASS_NAMES, BRATS_DISEASE_ACTIVITY_PROMPTS, BRATS_NUM_CLASSES
from src.data.brats_dataset import BraTSDataset
from src.models.brainiac_variants import BrainIACEncoder
from src.models.fusion_components import (
    CLIPTextEncoder,
    OptimizedCLIPTipAdapter,
    SimpleFusionModel,
)


# ==================== Per-class edit configurations ====================
#
# Each class has one MEANINGFUL edit (a token swap that should push the
# model AWAY from this class if text prototypes are causally involved)
# and one CONTROL edit (a semantically null swap of comparable surface
# magnitude). The two are run side by side and compared.

INTERVENTION_CONFIG = {
    "Quiescent (minimal active disease)": {
        "meaningful": {
            "find": "stable",
            "replace": "unstable",
            "explanation": "Flip 'stable resection cavity' to 'unstable' "
                           "(should push toward an active-disease class).",
        },
        "control": {
            "find": "Post-operative",
            "replace": "Postoperative",
            "explanation": "Remove the hyphen — same meaning, surface-level edit.",
        },
    },

    "Enhancing without necrosis": {
        "meaningful": {
            "find": "enhancing",
            "replace": "non-enhancing",
            "explanation": "Negate the discriminating clinical feature; "
                           "should push toward Quiescent.",
        },
        "control": {
            "find": "Nodular",
            "replace": "Focal",
            "explanation": "Clinical synonym — same anatomical descriptor.",
        },
    },

    "Necrotic enhancing (ring-enhancing)": {
        "meaningful": {
            "find": "necrotic",
            "replace": "cystic",
            "explanation": "Replace 'necrotic' with 'cystic' — clinically "
                           "distinct fluid-filled appearance, should reduce "
                           "high-grade signal.",
        },
        "control": {
            "find": "Aggressive",
            "replace": "Invasive",
            "explanation": "Synonym in glioma reporting — both describe "
                           "spread, no change in clinical signal.",
        },
    },
}


# ==================== Prompt editing ====================

def edit_brats_prompts(class_name: str, find: str, replace: str) -> dict:
    """Return a copy of BRATS_DISEASE_ACTIVITY_PROMPTS with one token edited."""
    if class_name not in BRATS_DISEASE_ACTIVITY_PROMPTS:
        raise ValueError(
            f"Unknown BraTS class: {class_name!r}. "
            f"Valid: {list(BRATS_DISEASE_ACTIVITY_PROMPTS)}"
        )
    new_prompts = copy.deepcopy(BRATS_DISEASE_ACTIVITY_PROMPTS)
    original = new_prompts[class_name]
    edited = [p.replace(find, replace) for p in original]
    n_changed = sum(1 for o, e in zip(original, edited) if o != e)
    if n_changed == 0:
        raise ValueError(
            f"Token {find!r} not found in any prompt for {class_name!r}. "
            f"Check the prompts in src/config/constants.py."
        )
    new_prompts[class_name] = edited
    print(f"    Edited {n_changed}/{len(original)} prompts for {class_name!r} "
          f"({find!r} -> {replace!r})")
    return new_prompts


# ==================== Prototype buffer helpers ====================

def _get_text_prototypes_buffer(model) -> torch.Tensor:
    if not hasattr(model, "clip_branch") or model.clip_branch is None:
        raise RuntimeError("model.clip_branch is None - CLIP branch not initialized.")
    if hasattr(model.clip_branch, "text_prototypes"):
        return model.clip_branch.text_prototypes
    raise RuntimeError("text_prototypes buffer not found on model.clip_branch.")


def _set_text_prototypes_buffer(model, new_protos: torch.Tensor) -> None:
    buf = _get_text_prototypes_buffer(model)
    if buf.shape != new_protos.shape:
        raise ValueError(f"Shape mismatch when swapping prototypes: "
                         f"buffer {tuple(buf.shape)} vs new {tuple(new_protos.shape)}")
    buf.copy_(new_protos.to(buf.device))


# ==================== Inference helper ====================

@torch.no_grad()
def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (softmax probabilities, ground-truth labels) over the loader."""
    model.eval()
    probs_chunks: list[np.ndarray] = []
    labels_chunks: list[int] = []
    for volumes, labels in loader:
        volumes = volumes.to(device)
        out = model(volumes, mode="eval")
        logits = out[0] if isinstance(out, tuple) else out
        probs_chunks.append(F.softmax(logits, dim=-1).cpu().numpy())
        labels_chunks.extend(labels.tolist())
    return np.concatenate(probs_chunks, axis=0), np.array(labels_chunks)


# ==================== Text encoder prompt patching ====================

def patch_text_encoder_for_brats(text_encoder: CLIPTextEncoder) -> None:
    """Same monkey-patch used at training time: make the encoder default
    to BRATS_DISEASE_ACTIVITY_PROMPTS rather than the Kaggle ones."""
    import types
    original_build = text_encoder.build_text_prototypes

    def _brats_build(self, prompts_dict=None, device=None):
        return original_build(prompts_dict or BRATS_DISEASE_ACTIVITY_PROMPTS, device)

    text_encoder.build_text_prototypes = types.MethodType(_brats_build, text_encoder)


# ==================== Model reconstruction from checkpoint ====================

def rebuild_model_from_checkpoint(
    checkpoint_path: str,
    brainiac_weights_path: Optional[str],
    device: torch.device,
) -> tuple[SimpleFusionModel, CLIPTextEncoder]:
    """Reconstruct the trained fusion model with the right architecture, then
    load the saved state_dict (which includes the cache, prototypes, and weights).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"]

    # Recover cache shape from the saved state to instantiate the adapter
    cache_keys_key = next(
        k for k in state.keys() if k.endswith("tip_adapter.cache_keys")
    )
    cache_values_key = next(
        k for k in state.keys() if k.endswith("tip_adapter.cache_values")
    )
    cache_keys_shape = tuple(state[cache_keys_key].shape)
    cache_values_shape = tuple(state[cache_values_key].shape)
    print(f"  Saved cache shape: keys {cache_keys_shape}, values {cache_values_shape}")

    # Build encoders
    image_encoder = BrainIACEncoder(embed_dim=512, dropout=0.1)
    text_encoder = CLIPTextEncoder(class_names=BRATS_CLASS_NAMES)
    patch_text_encoder_for_brats(text_encoder)

    if brainiac_weights_path and Path(brainiac_weights_path).exists():
        try:
            image_encoder.load_pretrained_weights(brainiac_weights_path)
        except Exception as e:
            print(f"  WARN: BrainIAC weight preload failed: {e}")

    image_encoder = image_encoder.to(device)
    text_encoder = text_encoder.to(device)

    # Build fusion model
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
    model.share_backbone_with(image_encoder)

    # Manually attach the CLIP branch with dummy buffers of the right shape;
    # they'll be overwritten by load_state_dict below.
    dummy_keys = torch.zeros(cache_keys_shape, device=device)
    dummy_values = torch.zeros(cache_values_shape, device=device)
    model.clip_branch = OptimizedCLIPTipAdapter(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        cache_keys=dummy_keys,
        cache_values=dummy_values,
        alpha=BEST_CLIP_CONFIG["alpha"],
        t_knn=BEST_CLIP_CONFIG["t_knn"],
        lr_adapter=BEST_CLIP_CONFIG["lr_adapter"],
        device=device,
    ).to(device)
    model.image_encoder = image_encoder
    model.text_encoder = text_encoder
    model.device = device

    # Now load the trained weights + cache + prototypes
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARN: missing keys in state_dict: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  WARN: unexpected keys in state_dict: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    model.eval()

    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_acc={ckpt.get('val_acc', float('nan')):.4f})")
    return model, text_encoder


# ==================== Single-edit intervention ====================

def run_single_edit(
    model: SimpleFusionModel,
    text_encoder: CLIPTextEncoder,
    test_loader,
    device,
    baseline_probs: np.ndarray,
    true_labels: np.ndarray,
    class_name: str,
    edit_label: str,
    find: str,
    replace: str,
) -> dict:
    """Run one (meaningful or control) edit for one class. Returns a result dict."""
    class_idx = BRATS_CLASS_NAMES.index(class_name)

    # True positives at baseline = correctly predicted as this class
    baseline_preds = baseline_probs.argmax(axis=-1)
    positive_mask = (baseline_preds == class_idx) & (true_labels == class_idx)
    n_pos = int(positive_mask.sum())

    if n_pos == 0:
        print(f"    [{edit_label}] no baseline true positives for {class_name!r}; skipping.")
        return {
            "class": class_name,
            "edit_label": edit_label,
            "find": find,
            "replace": replace,
            "n_positives": 0,
            "mean_prob_drop": None,
            "median_prob_drop": None,
            "fraction_flipped": None,
        }

    edited_prompts = edit_brats_prompts(class_name, find, replace)
    new_protos = text_encoder.build_text_prototypes(edited_prompts, device=device)

    original_protos = _get_text_prototypes_buffer(model).clone()
    try:
        _set_text_prototypes_buffer(model, new_protos)
        edited_probs, _ = predict_proba(model, test_loader, device)
    finally:
        _set_text_prototypes_buffer(model, original_protos)

    drops = baseline_probs[positive_mask, class_idx] - edited_probs[positive_mask, class_idx]
    edited_pred_at_positives = edited_probs[positive_mask].argmax(axis=-1)
    flipped = float((edited_pred_at_positives != class_idx).mean())

    return {
        "class": class_name,
        "edit_label": edit_label,
        "find": find,
        "replace": replace,
        "n_positives": n_pos,
        "mean_prob_drop": float(drops.mean()),
        "median_prob_drop": float(np.median(drops)),
        "std_prob_drop": float(drops.std()),
        "fraction_flipped": flipped,
    }


# ==================== Sweep ====================

def run_brats_intervention_sweep(
    model: SimpleFusionModel,
    text_encoder: CLIPTextEncoder,
    test_loader,
    device,
) -> dict:
    """Run meaningful + control intervention on every BraTS class."""
    print("\n=== Baseline predictions ===")
    baseline_probs, true_labels = predict_proba(model, test_loader, device)
    baseline_preds = baseline_probs.argmax(axis=-1)
    overall_acc = float((baseline_preds == true_labels).mean())
    print(f"  Baseline test accuracy: {overall_acc:.4f} on {len(true_labels)} cases")

    per_class_tp = {}
    for c_idx, c_name in enumerate(BRATS_CLASS_NAMES):
        n_tp = int(((baseline_preds == c_idx) & (true_labels == c_idx)).sum())
        per_class_tp[c_name] = n_tp
        print(f"  True positives for {c_name!r}: {n_tp}")

    results = []
    for class_name, edit_pair in INTERVENTION_CONFIG.items():
        print(f"\n--- Class: {class_name} ---")
        for edit_label in ("meaningful", "control"):
            edit = edit_pair[edit_label]
            print(f"  [{edit_label}] {edit['find']!r} -> {edit['replace']!r}")
            r = run_single_edit(
                model, text_encoder, test_loader, device,
                baseline_probs, true_labels,
                class_name=class_name,
                edit_label=edit_label,
                find=edit["find"],
                replace=edit["replace"],
            )
            r["explanation"] = edit["explanation"]
            if r["mean_prob_drop"] is not None:
                print(f"    mean_drop={r['mean_prob_drop']:+.4f}  "
                      f"median_drop={r['median_prob_drop']:+.4f}  "
                      f"flipped={r['fraction_flipped']:.2%}")
            results.append(r)

    summary = {
        "baseline_accuracy": overall_acc,
        "per_class_true_positives": per_class_tp,
        "n_test_cases": int(len(true_labels)),
        "edits": results,
    }
    return summary


# ==================== Faithfulness reporting ====================

def print_faithfulness_table(summary: dict) -> None:
    print("\n========== Faithfulness comparison ==========")
    print(f"{'Class':<42s} {'Meaningful Δp':>16s} {'Meaningful flip%':>20s} "
          f"{'Control Δp':>16s} {'Control flip%':>16s}")

    edits_by_class = {}
    for r in summary["edits"]:
        edits_by_class.setdefault(r["class"], {})[r["edit_label"]] = r

    for class_name, by_label in edits_by_class.items():
        m = by_label.get("meaningful", {})
        c = by_label.get("control", {})
        m_drop = m.get("mean_prob_drop")
        m_flip = m.get("fraction_flipped")
        c_drop = c.get("mean_prob_drop")
        c_flip = c.get("fraction_flipped")

        def fmt_drop(x):
            return f"{x:+.4f}" if x is not None else "n/a"

        def fmt_flip(x):
            return f"{100 * x:.2f}%" if x is not None else "n/a"

        print(f"{class_name:<42s} {fmt_drop(m_drop):>16s} {fmt_flip(m_flip):>20s} "
              f"{fmt_drop(c_drop):>16s} {fmt_flip(c_flip):>16s}")

    print("\nA faithful interpretability mechanism implies meaningful drop >> control drop.")
    print("If they are comparable, the prototypes act as labels rather than concepts.")


# ==================== Main ====================

def main():
    p = argparse.ArgumentParser(
        description="BraTS concept-intervention falsification sweep.",
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to best.pt from train_fusion_brats.py.")
    p.add_argument("--manifest", required=True,
                   help="manifest_brats_train.csv produced by inventory_brats.py.")
    p.add_argument("--brainiac_weights", default="weights/brainiac/BrainIAC.ckpt",
                   help="Optional preload of BrainIAC weights "
                        "(overwritten by checkpoint, but useful for initialization).")
    p.add_argument("--modality", default="t2f",
                   choices=["t1n", "t1c", "t2w", "t2f"])
    p.add_argument("--target_size", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test"],
                   help="Which manifest split to run the intervention against. "
                        "Default 'test' for honest reporting.")
    p.add_argument("--out_json", default=None,
                   help="Optional path to save the result JSON.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"manifest:   {args.manifest}")
    print(f"split:      {args.split}")

    # ---- Build the test dataset / loader ----------------------------
    test_ds = BraTSDataset(
        manifest_csv=args.manifest,
        split=args.split,
        modality=args.modality,
        target_size=args.target_size,
        multimodal=False,
        augment=False,
        require_label=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"  {args.split} loader: {len(test_ds)} patients")

    # ---- Reconstruct trained model ----------------------------------
    print("\n=== Rebuilding trained fusion model ===")
    model, text_encoder = rebuild_model_from_checkpoint(
        args.checkpoint, args.brainiac_weights, device,
    )

    # ---- Run sweep --------------------------------------------------
    summary = run_brats_intervention_sweep(model, text_encoder, test_loader, device)
    summary["checkpoint"] = args.checkpoint
    summary["split"] = args.split
    summary["modality"] = args.modality

    print_faithfulness_table(summary)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
