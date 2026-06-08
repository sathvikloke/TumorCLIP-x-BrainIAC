# TumorCLIP × BrainIAC

Extension of Zongyu Li et al.'s [TumorCLIP](https://doi.org/10.64898/2026.03.11.26348155) (ISMRM 2026) with a [BrainIAC](https://github.com/AIM-KannLab/BrainIAC) backbone (Tak et al., *Nature Neuroscience* 2026) dropped in where DenseNet121 used to be. Adds a concept-intervention falsification test for the interpretability claim, and provides both Jupyter notebook and CLI workflows.

**For full setup, training instructions, and the concept-intervention test, see [`BRAINIAC_README.md`](./BRAINIAC_README.md).**

## What's in here

| Path | What |
| --- | --- |
| `train_fusion.py` | CLI entry point for BrainIAC + CLIP fusion training |
| `train_single_modal.py` | CLI entry point for stage-1 BrainIAC head fine-tuning |
| `BrainIAC_CLIP_Fusion_Model_Training.ipynb` | Notebook version of the fusion training |
| `BrainIAC_Enhanced_Single_Modal_Training.ipynb` | Notebook version of stage-1 training |
| `scripts/concept_intervention.py` | Phase 5 interpretability-falsification test |
| `src/models/brainiac_variants.py` | BrainIAC 3-D ViT wrapper (direct port of Tak et al.'s `src/model.py`) |
| `src/models/fusion_components.py` | `CLIPTextEncoder` + `TipAdapter` + `SimpleFusionModel` (ported from Zongyu's notebook) |
| `src/models/densenet_variants.py`, `src/models/single_modal/`, `src/training/`, `src/data/`, `src/config/`, `src/core/`, `src/visualization/` | Zongyu's original TumorCLIP source code, unchanged |
| `CLIP_Fusion_Model_Training.ipynb`, `Enhanced_Single_Modal_Training.ipynb`, `Model_Evaluation_Visualization.ipynb` | Zongyu's original notebooks (preserved for reference / DenseNet comparison) |

## Quick start

```bash
pip install -r requirements.txt
# Place BrainIAC weights at weights/brainiac/BrainIAC.ckpt
# Place Kaggle data at data/train/<class>/ and data/test/<class>/
python train_fusion.py --data_root data --brainiac_weights weights/brainiac/BrainIAC.ckpt --epochs 15 --batch_size 8
```

Pass `--help` to either script for all available flags. See [`BRAINIAC_README.md`](./BRAINIAC_README.md) for the full setup walkthrough, the source-vs-original audit of every line in `brainiac_variants.py`, and the concept-intervention experiment protocol.

## Honest caveats

- The 2-D Kaggle slices are inflated to thin 96³ "volumes" before being fed to BrainIAC's 3-D ViT. BrainIAC was pretrained on real 3-D MRI; running it on thin-volume inputs is a Phase-1 compromise. For the genuinely informative comparison, use 3-D datasets like BraTS, UPenn-GBM, or UCSF-PDGM.
- Don't commit BrainIAC's `.ckpt` to this repo. The `.gitignore` keeps `weights/`, `data/`, and `results/` out by default.

## Credits

- TumorCLIP architecture and code: Jia Y., Niu J., Li Z., Guo J. *TumorCLIP: Lightweight Vision–Language Fusion for Explainable MRI-Based Brain Tumor Classification.* ISMRM 2026, Abstract 401-02-001. doi: 10.64898/2026.03.11.26348155
- BrainIAC architecture and pretrained weights: Tak D., Garomsa B. A., Zapaishchykova A., et al. *A generalizable foundation model for analysis of human brain MRI.* Nature Neuroscience 2026.
- This extension: [sathvikloke](https://github.com/sathvikloke), Albert Einstein College of Medicine.
