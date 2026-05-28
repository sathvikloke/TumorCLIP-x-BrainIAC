# TumorCLIP Reimplementation

Faithful reimplementation of TumorCLIP (Jia et al., ISMRM 2026, *"TumorCLIP: Lightweight Vision-Language Fusion for Explainable MRI-Based Brain Tumor Classification"*).

This is a clean baseline meant to serve Phase 1 of the TumorCLIP × BrainIAC project. The architecture is designed so the DenseNet121 backbone can be swapped for a foundation model (BrainIAC, UMBIF, Decipher-MR) with a single config change in Phase 2.

## What it implements

- DenseNet121 backbone (ImageNet-pretrained, fine-tuned on tumor data)
- Frozen CLIP text encoder producing class-level text prototypes
- Tip-Adapter-style late fusion combining image-only logits with text-prototype cosine-similarity logits
- Fusion weight α = 0.3 (paper's reported optimum)
- 6-class classification: Glioma, Meningioma, Normal, Neurocytoma, Other Lesions, Schwannoma
- Dataset loader for the Kaggle "Brain Tumor MRI Images (17 Classes)" dataset with the 17→6 superclass mapping
- Training loop with Adam, CosineAnnealingLR, early stopping, W&B logging

## Setup

```bash
conda create -n tumorclip python=3.10 -y
conda activate tumorclip
pip install -r requirements.txt
```

## Download the data

```bash
# Get a Kaggle API key from https://www.kaggle.com/settings, save to ~/.kaggle/kaggle.json
bash scripts/download_data.sh
```

This pulls the dataset to `data/raw/` and runs the 17→6 consolidation script, producing `data/processed/train/` and `data/processed/test/`.

## Verify the install (30 seconds)

Before downloading the dataset, confirm the code works end-to-end with synthetic data:

```bash
python -m tests.test_smoke
```

This builds the prototype bank, runs a forward pass, builds a fake dataset, runs `evaluate()`, and runs `concept_intervention()`. If all 5 steps print PASS, every code path is wired up correctly. (This downloads ~150 MB of CLIP and DenseNet weights on first run.)

## Train the baseline

```bash
python -m src.train --config configs/baseline.yaml
```

Expected result on test split: ~97.5–98.5% accuracy (paper reports 98.5%). If you're within 2pp, replication is confirmed.

## Project layout

```
tumorclip_repro/
├── README.md
├── requirements.txt
├── configs/
│   └── baseline.yaml          # DenseNet121 + TumorCLIP head config
├── scripts/
│   ├── download_data.sh       # Kaggle CLI download + 17→6 mapping
│   └── consolidate_classes.py # 17→6 superclass mapping
└── src/
    ├── __init__.py
    ├── prototypes.py          # Class descriptions + CLIP text encoding
    ├── data.py                # PyTorch Dataset for processed Kaggle data
    ├── model.py               # TumorCLIP module
    ├── train.py               # Training loop
    └── evaluate.py            # Metrics: accuracy, macro-F1, per-class recall
```

## Phase 2 hooks (BrainIAC swap)

The model is split into a `Backbone` class and a `TumorCLIPHead` class so you can swap the backbone without touching the head:

```python
from src.model import TumorCLIPHead
from brainiac import BrainIACBackbone  # to be added in Phase 2

backbone = BrainIACBackbone(pretrained=True, freeze=True)
head = TumorCLIPHead(in_features=2048, n_classes=6, alpha=0.3)
```

The text-prototype generation in `prototypes.py` does not depend on the backbone and works identically across configs.

## Concept intervention (Phase 5)

The text prototypes are loaded from `prototypes.py` as editable strings. To run a concept intervention experiment, edit a single token and re-run inference on the test set. See `prototypes.py` docstring for the protocol.

## Citation

If you use this reimplementation, please cite the original TumorCLIP abstract:

> Jia Y., Niu J., Li Z., Guo J. *TumorCLIP: Lightweight Vision-Language Fusion for Explainable MRI-Based Brain Tumor Classification.* ISMRM 2026, Abstract 401-02-001.

## Status

This is an *unofficial* reimplementation based on the published abstract. No code was released by the original authors as of [today's date]. Authors contacted at `zl3372@columbia.edu`.
