"""Phase 5: concept-intervention falsification test.

Tests whether the BrainIAC + CLIP fusion model's interpretability claim is
*causal* (the model actually uses the concepts) or *decorative* (the
prototypes function as labels).

Because the fusion model itself is defined inline in
``BrainIAC_CLIP_Fusion_Model_Training.ipynb`` (mirroring Zongyu's structure),
this script is meant to be run **as a notebook cell** at the end of that
notebook, after a trained model is in memory. The functions below take the
already-instantiated ``model`` (a SimpleFusionModel with BrainIAC backbone)
and ``text_encoder`` (a CLIPTextEncoder) from the notebook's namespace and
run the intervention against them.

Usage from a notebook cell::

    from scripts.concept_intervention import run_intervention
    result = run_intervention(
        model=fusion_model,
        text_encoder=text_encoder,
        loader=test_loader,
        device=device,
        class_name="Glioma",
        find="ring-enhancing",
        replace="non-enhancing",
        control_find="tumor",
        control_replace="object",
    )

A meaningful interpretability claim means the meaningful edit produces
substantially larger probability drops / prediction flips than the control
edit. If they're comparable, the prototypes act as labels, not concepts.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Allow importing from src/ when invoked as `python -m scripts.concept_intervention`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.constants import CLASS_NAMES, PROFESSIONAL_MEDICAL_PROMPTS


# ---------------------------------------------------------------------------
# Prompt editing
# ---------------------------------------------------------------------------

def edit_prompts(class_name: str, find: str, replace: str) -> dict:
    """Return a copy of PROFESSIONAL_MEDICAL_PROMPTS with one token edited
    in every prompt for ``class_name`` that contains it.
    """
    if class_name not in PROFESSIONAL_MEDICAL_PROMPTS:
        raise ValueError(f"Unknown class: {class_name!r}. Choose from {CLASS_NAMES}")
    new_prompts = copy.deepcopy(PROFESSIONAL_MEDICAL_PROMPTS)
    original = new_prompts[class_name]
    edited = [p.replace(find, replace) for p in original]
    n_changed = sum(1 for o, e in zip(original, edited) if o != e)
    if n_changed == 0:
        raise ValueError(
            f"Token {find!r} not found in any prompt for {class_name!r}. "
            f"Check spelling. Prompts: {original}"
        )
    new_prompts[class_name] = edited
    print(f"  Edited {n_changed}/{len(original)} prompts for {class_name}")
    return new_prompts


# ---------------------------------------------------------------------------
# Prediction helpers (works with Zongyu's SimpleFusionModel structure)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probs, labels)`` from a SimpleFusionModel-style ``model``.

    Handles both the tuple return of the fusion path
    (``fused_logits, brainiac_logits, clip_logits``) and the single-tensor
    return of the BrainIAC-only path.
    """
    model.eval()
    probs_list: list[np.ndarray] = []
    labels_list: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        out = model(images, mode="eval")
        logits = out[0] if isinstance(out, tuple) else out
        probs_list.append(F.softmax(logits, dim=-1).cpu().numpy())
        labels_list.extend(labels.tolist())
    return np.concatenate(probs_list, axis=0), np.array(labels_list)


def _get_text_prototypes_buffer(model) -> torch.Tensor:
    """Locate the text_prototypes buffer inside Zongyu's nested fusion model.

    Path in SimpleFusionModel: model.clip_branch.text_prototypes
    (OptimizedCLIPTipAdapter holds the buffer).
    """
    if not hasattr(model, "clip_branch") or model.clip_branch is None:
        raise RuntimeError("model.clip_branch is None — CLIP branch wasn't set up.")
    if hasattr(model.clip_branch, "text_prototypes"):
        return model.clip_branch.text_prototypes
    raise RuntimeError("Could not find text_prototypes on model.clip_branch.")


def _set_text_prototypes_buffer(model, new_protos: torch.Tensor) -> None:
    """In-place swap of text_prototypes (same shape required)."""
    buf = _get_text_prototypes_buffer(model)
    if buf.shape != new_protos.shape:
        raise ValueError(f"Shape mismatch: {buf.shape} vs {new_protos.shape}")
    buf.copy_(new_protos.to(buf.device))


# ---------------------------------------------------------------------------
# One intervention
# ---------------------------------------------------------------------------

def run_intervention(
    model,
    text_encoder,
    loader,
    device,
    class_name: str,
    find: str,
    replace: str,
    control_find: Optional[str] = None,
    control_replace: Optional[str] = None,
    save_to: Optional[str] = None,
) -> list[dict]:
    """Run the meaningful edit (and optional control edit) on one class.

    Args:
        model:        a trained SimpleFusionModel (Zongyu-style) with a CLIP branch.
        text_encoder: a CLIPTextEncoder exposing ``build_text_prototypes(prompts_dict, device=...)``.
        loader:       test DataLoader.
        device:       'cuda' or 'cpu'.
        class_name:   one of CLASS_NAMES.
        find/replace: the meaningful token edit.
        control_find/control_replace: optional control edit; faithful interpretability
                                       implies the control moves predictions much less.
        save_to:      optional JSON output path.

    Returns:
        list of result dicts (1 for meaningful, +1 if control specified).
    """
    edits = [(find, replace, "meaningful")]
    if control_find:
        if not control_replace:
            raise ValueError("control_replace is required when control_find is given.")
        edits.append((control_find, control_replace, "control"))

    class_idx = CLASS_NAMES.index(class_name)

    print(f"\nBaseline predictions on {len(loader.dataset)} test cases…")
    baseline_probs, true_labels = predict_proba(model, loader, device)
    baseline_preds = baseline_probs.argmax(axis=-1)

    positive_mask = (baseline_preds == class_idx) & (true_labels == class_idx)
    n_pos = int(positive_mask.sum())
    print(f"  true positives for {class_name!r}: {n_pos}")
    if n_pos == 0:
        print("  no positives — intervention is undefined for this class on this run.")
        return []

    # Snapshot original prototypes so we can restore after each edit
    original_protos = _get_text_prototypes_buffer(model).clone()

    results = []
    for find_tok, replace_tok, label in edits:
        print(f"\n[{label}] {class_name}: {find_tok!r} -> {replace_tok!r}")
        edited_prompts = edit_prompts(class_name, find_tok, replace_tok)
        new_protos = text_encoder.build_text_prototypes(edited_prompts, device=device)

        try:
            _set_text_prototypes_buffer(model, new_protos)
            edited_probs, _ = predict_proba(model, loader, device)
        finally:
            _set_text_prototypes_buffer(model, original_protos)

        drops = baseline_probs[positive_mask, class_idx] - edited_probs[positive_mask, class_idx]
        edited_preds_pos = edited_probs[positive_mask].argmax(axis=-1)
        flipped = float((edited_preds_pos != class_idx).mean())

        result = {
            "label": label,
            "class": class_name,
            "find": find_tok,
            "replace": replace_tok,
            "n_positives": n_pos,
            "mean_prob_drop": float(drops.mean()),
            "median_prob_drop": float(np.median(drops)),
            "fraction_flipped": flipped,
        }
        print(
            f"  mean_drop={result['mean_prob_drop']:+.4f}  "
            f"median_drop={result['median_prob_drop']:+.4f}  "
            f"flipped={result['fraction_flipped']:.2%}"
        )
        results.append(result)

    if len(results) == 2:
        meaningful, control = results
        print("\nFaithfulness comparison:")
        print(f"  Meaningful edit mean drop: {meaningful['mean_prob_drop']:+.4f}  "
              f"flipped: {meaningful['fraction_flipped']:.2%}")
        print(f"  Control edit    mean drop: {control['mean_prob_drop']:+.4f}  "
              f"flipped: {control['fraction_flipped']:.2%}")
        print(
            "\nFaithful interpretability ⇒ meaningful drop >> control drop. "
            "If they're comparable, the prototypes act as labels, not concepts."
        )

    if save_to:
        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point (if you want to script it from a checkpoint)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        "This script is meant to be imported from inside the BrainIAC fusion "
        "notebook (run_intervention takes the trained `model` and `text_encoder` "
        "from the notebook's namespace). To run from a checkpoint you'd need to "
        "rebuild the full SimpleFusionModel + CLIPTextEncoder in a script — see "
        "BRAINIAC_README.md for the recipe."
    )
