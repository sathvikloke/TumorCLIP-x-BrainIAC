"""
Evaluation metrics and the Phase 5 concept-intervention protocol.

Two entry points:

* `evaluate(model, loader, device)` — accuracy, macro-F1, per-class recall, confusion matrix.
* `concept_intervention(model, loader, device, find, replace)` — measures
  faithfulness of the text-prototype interpretability story by editing one
  concept token and rerunning inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader

from .prototypes import CLASS_DESCRIPTIONS, CLASS_NAMES, build_prototype_bank, edit_description


@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    per_class_recall: dict[str, float]
    confusion: np.ndarray
    report: str


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> EvalResult:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in loader:
        # labels stay on CPU — we only use them for sklearn metrics
        images = images.to(device)
        out = model(images)
        preds = out["logits"].argmax(dim=-1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    per_class = recall_score(
        y_true_np, y_pred_np, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    per_class_dict = {c: float(per_class[i]) for i, c in enumerate(CLASS_NAMES)}

    return EvalResult(
        accuracy=float(accuracy_score(y_true_np, y_pred_np)),
        macro_f1=float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        per_class_recall=per_class_dict,
        confusion=confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(CLASS_NAMES)))),
        report=classification_report(
            y_true_np, y_pred_np, target_names=CLASS_NAMES, zero_division=0
        ),
    )


@torch.no_grad()
def predict_proba(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (n, C) softmax probabilities and (n,) true labels."""
    model.eval()
    probs_list: list[np.ndarray] = []
    labels_list: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        out = model(images)
        probs = F.softmax(out["logits"], dim=-1)
        probs_list.append(probs.cpu().numpy())
        labels_list.extend(labels.cpu().tolist())
    return np.concatenate(probs_list, axis=0), np.array(labels_list)


@dataclass
class InterventionResult:
    class_name: str
    find: str
    replace: str
    mean_prob_drop: float           # mean drop in target class probability across positives
    median_prob_drop: float
    n_positives: int                # how many test cases were true positives for this class
    fraction_flipped: float         # fraction whose argmax prediction changed after the edit


def concept_intervention(
    model,
    loader: DataLoader,
    device: torch.device,
    class_name: str,
    find: str,
    replace: str,
    text_encoder_name: str = "ViT-B-32",
    text_encoder_pretrained: str = "openai",
) -> InterventionResult:
    """Run the Phase 5 falsification test for one concept edit.

    Steps:
      1. Get baseline probabilities and predictions with current prototypes.
      2. Edit `find -> replace` inside CLASS_DESCRIPTIONS[class_name].
      3. Rebuild the prototype bank with the edited description.
      4. Swap the new prototypes into the model.
      5. Recompute probabilities and predictions.
      6. Measure (a) drop in target-class probability on the cases that were
         correctly predicted as `class_name` originally, and (b) fraction
         whose argmax flipped.

    A meaningful interpretability claim implies a measurable drop and/or
    flip rate. No change implies the prototypes act as labels, not concepts.
    """
    if class_name not in CLASS_DESCRIPTIONS:
        raise ValueError(f"Unknown class: {class_name!r}")

    class_idx = CLASS_NAMES.index(class_name)
    baseline_probs, true_labels = predict_proba(model, loader, device)
    baseline_preds = baseline_probs.argmax(axis=1)

    # Only measure cases the model originally predicted as `class_name` and got right.
    positive_mask = (baseline_preds == class_idx) & (true_labels == class_idx)
    n_positives = int(positive_mask.sum())
    if n_positives == 0:
        raise RuntimeError(
            f"No true-positive cases for {class_name!r}; intervention is ill-defined."
        )

    # Build counterfactual prototypes
    edited_descriptions = dict(CLASS_DESCRIPTIONS)
    edited_descriptions[class_name] = edit_description(
        CLASS_DESCRIPTIONS[class_name], find, replace
    )
    new_bank = build_prototype_bank(
        model_name=text_encoder_name,
        pretrained=text_encoder_pretrained,
        descriptions=edited_descriptions,
        device=device,
    )

    # Swap prototypes into the model (the head holds them as a buffer).
    # The clone is outside the try (it cannot fail interestingly); everything
    # that *could* leave the buffer in a bad state goes inside.
    original_protos = model.head.text_prototypes.clone()
    try:
        model.head.text_prototypes.copy_(new_bank.embeddings.to(device))
        edited_probs, _ = predict_proba(model, loader, device)
    finally:
        # Always restore original prototypes, even if the swap or inference failed.
        model.head.text_prototypes.copy_(original_protos)

    edited_preds = edited_probs.argmax(axis=1)
    prob_drops = baseline_probs[positive_mask, class_idx] - edited_probs[positive_mask, class_idx]
    fraction_flipped = float((edited_preds[positive_mask] != class_idx).mean())

    return InterventionResult(
        class_name=class_name,
        find=find,
        replace=replace,
        mean_prob_drop=float(prob_drops.mean()),
        median_prob_drop=float(np.median(prob_drops)),
        n_positives=n_positives,
        fraction_flipped=fraction_flipped,
    )
