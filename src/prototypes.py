"""
Text prototypes for TumorCLIP.

The paper describes "frozen text prototypes of class descriptions" via a CLIP-style
text encoder. The original paper does not publish the exact strings used; the
descriptions below are clinically grounded placeholders that match the example
quoted in the paper ("extra-axial mass with dural tail" for meningioma).

For Phase 5 concept-intervention experiments, edit a single token in any
description string, regenerate prototypes, and rerun inference. The faithfulness
test in `evaluate.py` automates this.

CLASS_DESCRIPTIONS is the canonical mapping; edit here, not at call sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import open_clip
import torch
import torch.nn.functional as F


# Class indices used throughout the codebase. The order here is the order of the
# output logits.
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "Normal",
    "Neurocytoma",
    "Other Lesions",
    "Schwannoma",
]

# Clinically grounded text descriptions, one per class. These follow the
# style cued in the TumorCLIP paper ("extra-axial mass with dural tail" for
# meningioma) and are intentionally written with concept tokens that can be
# edited in the Phase 5 intervention experiment (e.g. "ring-enhancing",
# "extra-axial", "dural tail", "intra-ventricular").
CLASS_DESCRIPTIONS: dict[str, str] = {
    "Glioma": (
        "An intra-axial heterogeneous brain mass with irregular ring enhancement, "
        "central necrosis, and surrounding T2/FLAIR hyperintense edema, "
        "consistent with a glioma."
    ),
    "Meningioma": (
        "An extra-axial dural-based mass with homogeneous contrast enhancement "
        "and a characteristic dural tail sign, consistent with a meningioma."
    ),
    "Normal": (
        "Brain MRI with no abnormal mass, lesion, or pathological signal "
        "abnormality identified; normal-appearing brain parenchyma."
    ),
    "Neurocytoma": (
        "A well-circumscribed intra-ventricular mass within the lateral ventricle, "
        "often attached to the septum pellucidum, with heterogeneous signal and "
        "scattered cystic components, consistent with a neurocytoma."
    ),
    "Other Lesions": (
        "Other brain lesions including miscellaneous focal abnormalities not "
        "classified as glioma, meningioma, neurocytoma, or schwannoma."
    ),
    "Schwannoma": (
        "A well-circumscribed extra-axial mass arising from a cranial nerve, "
        "typically located along the cerebellopontine angle, with heterogeneous "
        "contrast enhancement, consistent with a schwannoma."
    ),
}


@dataclass
class PrototypeBank:
    """A frozen bank of text-prototype embeddings, one per class."""

    embeddings: torch.Tensor  # (n_classes, embed_dim), L2-normalized
    class_names: list[str]
    descriptions: list[str]

    def to(self, device: torch.device) -> "PrototypeBank":
        return PrototypeBank(
            embeddings=self.embeddings.to(device),
            class_names=self.class_names,
            descriptions=self.descriptions,
        )


def build_prototype_bank(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    descriptions: dict[str, str] | None = None,
    device: torch.device | str = "cpu",
) -> PrototypeBank:
    """Encode class descriptions with a frozen CLIP text encoder.

    Args:
        model_name: open_clip model name. Default: ViT-B-32.
        pretrained: open_clip pretrained tag. Default: openai.
        descriptions: optional override of {class_name: description}.
            Falls back to CLASS_DESCRIPTIONS.
        device: device on which to run the encoder.

    Returns:
        PrototypeBank with L2-normalized embeddings.
    """
    if descriptions is None:
        descriptions = CLASS_DESCRIPTIONS

    # Preserve canonical class ordering
    class_names = list(CLASS_NAMES)
    description_list = [descriptions[c] for c in class_names]

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    with torch.no_grad():
        tokens = tokenizer(description_list).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    # Move off device after encoding — we cache them once
    return PrototypeBank(
        embeddings=text_features.detach().cpu(),
        class_names=class_names,
        descriptions=description_list,
    )


def edit_description(
    original: str, find: str, replace: str
) -> str:
    """Edit a single concept token in a description, for intervention experiments.

    Example:
        edit_description(CLASS_DESCRIPTIONS["Glioma"], "ring enhancement", "no enhancement")

    Use with `build_prototype_bank(descriptions=...)` to rebuild a counterfactual
    prototype bank, then rerun model.predict() and measure the change in the
    target-class probability.
    """
    if find not in original:
        raise ValueError(f"Concept token {find!r} not found in description.")
    return original.replace(find, replace)
