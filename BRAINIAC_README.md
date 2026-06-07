# TumorCLIP × BrainIAC — line-for-line backbone swap

This repo is Zongyu Li et al.'s TumorCLIP, **unchanged**, with one extra component: BrainIAC's actual 3-D ViT backbone (Tak et al. 2026) dropped in where DenseNet121 used to be. The notebooks are programmatic clones of the originals — same cells, same logic, same hyperparameters — with only the backbone references renamed (`DenseNet → BrainIAC`, `densenet_branch → brainiac_branch`, etc.).

**What BrainIAC actually is** (verified against the official release at github.com/AIM-KannLab/BrainIAC, file `src/model.py`):

* **MONAI 3-D ViT-B** (not ResNet50): `in_channels=1`, `img_size=(96,96,96)`, `patch_size=(16,16,16)`, `hidden_size=768`, `mlp_dim=3072`, 12 layers, 12 heads
* **Feature output**: CLS token only, dimension 768
* **Checkpoint format**: Lightning `.ckpt` with `state_dict[…]`; keys prefixed with `"backbone."` (stripped at load time, per BrainIAC's `load_brainiac.py`)
* **Expected input preprocessing**: trilinear resize to 96³, single-channel, `NormalizeIntensityd(nonzero=True, channel_wise=True)` z-score

## What's added

| New file | What it is |
| --- | --- |
| `src/models/brainiac_variants.py` | Contains **three classes copied verbatim from BrainIAC `src/model.py`** (`ViTBackboneNet`, `Classifier`, `SingleScanModel`), plus two wrappers (`BrainIACClassifier`, `BrainIACEncoder`) that compose those classes with **the loss + optimizer + projection patterns copied verbatim from Zongyu's `densenet_variants.py`**. The only original code in the file is a 2-D→3-D adapter, required because nothing in either source bridges Zongyu's 2-D RGB inputs and BrainIAC's 3-D 96³ inputs. Provenance is annotated inline per method. |
| `BrainIAC_CLIP_Fusion_Model_Training.ipynb` | Programmatic clone of `CLIP_Fusion_Model_Training.ipynb` with `DenseNet → BrainIAC` substitutions applied to every cell. Same `SimpleFusionModel`, same `TipAdapter`, same multi-task loss (0.5 × CE + 0.3 × Focal + 0.2 × CE), same AdamW + cosine schedule, same multilingual prompts. |
| `BrainIAC_Enhanced_Single_Modal_Training.ipynb` | Clone of `Enhanced_Single_Modal_Training.ipynb` with `MODEL_NAMES` reduced to just `'BrainIAC'`. **Optional** — see "Two paths" below. |
| `scripts/concept_intervention.py` | Phase 5 falsification test. Import `run_intervention` from a notebook cell after training; edits a clinically-meaningful token in one class's prompts (e.g. `ring-enhancing → non-enhancing`), rebuilds prototypes, swaps them in place, measures probability drop and prediction flip rate. Supports a control edit. |

The three deprecated files (`src/models/brainiac_encoder.py`, `src/models/brainiac_fusion.py`, `scripts/train_brainiac_fusion.py`) raise on import — they were my earlier reimplementation and are kept only to make the directory's history obvious.

## Two ways to run

The same training pipeline is available as either Jupyter notebooks **or** plain Python scripts — pick whichever you prefer.

**Option 1: Python scripts (recommended for terminal / VS Code workflow).**

```
python train_single_modal.py --data_root data --brainiac_weights weights/brainiac/BrainIAC.ckpt --epochs 40
python train_fusion.py        --data_root data --brainiac_weights weights/brainiac/BrainIAC.ckpt --epochs 15
```

Each script has `--help` with all available flags. The fusion script auto-enables `--shared_backbone` (halves GPU memory). Pass `--no_shared_backbone` to revert to Zongyu's two-copies design.

**Option 2: Jupyter notebooks (preserves Zongyu's exact cell structure).**

Open `BrainIAC_Enhanced_Single_Modal_Training.ipynb` and `BrainIAC_CLIP_Fusion_Model_Training.ipynb` in VS Code or Jupyter, pick a kernel, and run cells top-to-bottom. Same logic as the scripts above — just cell-by-cell.

## Setup

```
pip install -r requirements.txt
pip install open-clip-torch monai
```

## Two paths to running

**Path A (recommended): skip stage 1, use BrainIAC's released weights directly.**

BrainIAC's whole point is that the 3D ResNet50 is already pretrained on ~49,000 brain MRIs — so the stage-1 DenseNet pretraining isn't needed. Download BrainIAC's checkpoint, point the fusion notebook at it, run:

```
mkdir -p results/best_models
cp /path/to/brainiac_resnet50_simclr.pth results/best_models/BrainIAC_Adam_lr0.0001_best.pth
jupyter notebook BrainIAC_CLIP_Fusion_Model_Training.ipynb
```

(Or edit `PRETRAINED_MODEL_PATH` in cell 4 of the notebook to point wherever you put it.)

**Path B: parallel stage 1.**

If you want to fine-tune BrainIAC's head on the tumor data first (à la Zongyu's DenseNet stage 1), run `BrainIAC_Enhanced_Single_Modal_Training.ipynb`. Registration with `ModelRegistry` is **automatic** — `brainiac_variants.py` registers `'BrainIAC'` (with batch_size=8) at import time, so the notebook just needs to import the module once early on. Then the existing grid-search infrastructure finds it.

The single-modal notebook produces `results/best_models/BrainIAC_Adam_lr0.0001_best.pth`. Then run the fusion notebook as above.

## Dataset

Same as Zongyu's original — `data/train/<class>/` and `data/test/<class>/` with the six classes `Glioma`, `Meningioma`, `NORMAL`, `Neurocitoma`, `Outros Tipos de Lesões`, `Schwannoma`. See the original `README.md` for details.

## Running the concept-intervention test

Inside `BrainIAC_CLIP_Fusion_Model_Training.ipynb`, after `fusion_model` is trained, add a cell:

```python
from scripts.concept_intervention import run_intervention

results = run_intervention(
    model=fusion_model,
    text_encoder=text_encoder,
    loader=test_loader,
    device=device,
    class_name="Glioma",
    find="ring-enhancing",
    replace="non-enhancing",
    control_find="tumor",
    control_replace="object",
    save_to="results/concept_intervention/glioma_ring_enhancing.json",
)
```

The output prints the meaningful-vs-control comparison and writes the per-case stats to JSON. A faithful interpretability claim means the meaningful edit moves predictions much more than the control. If they're comparable, the prototypes work as labels rather than concepts — also a publishable finding.

## Source-vs-original audit of `brainiac_variants.py`

| Component | Source | What it is |
| --- | --- | --- |
| `ViTBackboneNet` class | BrainIAC `src/model.py` lines 7–46 | **verbatim port** |
| MONAI `ViT(in_channels=1, img_size=(96,96,96), patch_size=(16,16,16), hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12, save_attn=True)` | BrainIAC `src/model.py` lines 12–21 | **verbatim port** |
| `torch.load(map_location="cpu", weights_only=False)` + `ckpt.get("state_dict", ckpt)` + `key[9:]` prefix strip + `strict=True` | BrainIAC `src/model.py` lines 23–37 | **verbatim port** |
| CLS-token forward `features[0][:, 0]` | BrainIAC `src/model.py` lines 40–46 | **verbatim port** |
| `Classifier(d_model=768, num_classes=…)` class with `nn.Linear` | BrainIAC `src/model.py` lines 48–54 | **verbatim port** |
| `SingleScanModel(backbone, classifier)` with `dropout(p=0.2)` between | BrainIAC `src/model.py` lines 56–66 | **verbatim port** |
| `BrainIACClassifier.__init__` signature (`backbone_lr=5e-5, head_lr=1e-3, focal_gamma=2.0, label_smoothing=0.05`) | Zongyu `densenet_variants.py::DenseNetClassifier` | **verbatim port** |
| `compute_loss(loss_type='focal'|'label_smooth'|'ce')` | Zongyu `densenet_variants.py::DenseNetClassifier.compute_loss` | **verbatim port** |
| `get_optimizer_params` returning backbone/head LR groups | Zongyu `densenet_variants.py::DenseNetClassifier.get_optimizer_params` | **verbatim port** |
| `BrainIACEncoder` `forward_features` semantics + `feature_projection` (Linear→ReLU→Dropout→Linear) + `new_classification_head` | Zongyu `densenet_variants.py::DenseNetEncoder` | **verbatim port** |
| `BrainIACEncoder.forward(return_features, use_original_classifier)` dispatch | Zongyu `densenet_variants.py::DenseNetEncoder.forward` | **verbatim port** |
| `_apply_brainiac_normalize_intensity` → wraps **`monai.transforms.NormalizeIntensity(nonzero=True, channel_wise=True)`** | The exact transform in BrainIAC `src/dataset.py::get_validation_transform` | **invokes MONAI directly** |
| `adapt_2d_to_brainiac` (RGB→grayscale + trilinear resize to 96³ + MONAI z-score) | *neither source has any code that bridges 2-D and 3-D* | **my code — unavoidable** |
| `BrainIACEncoder.original_classifier = Classifier(d_model=768, num_classes=…)` | DenseNetEncoder aliases DenseNet's built-in classifier; BrainIAC's SimCLR-pretrained ViT has no classifier head to alias | **my code — unavoidable**, but uses BrainIAC's `Classifier` class internally |
| Auto-registration with `ModelRegistry` | *neither source* | **my code — convenience** so Zongyu's grid-search trainer finds `'BrainIAC'` |

In short: every model-defining line came from one of the two source codebases. The only original code is the dimensionality bridge (2-D → 3-D) and one fresh `nn.Linear` head, both of which are structurally required and use the source classes wherever possible.

## Three honest caveats

1. **BrainIAC was pretrained on 3-D 96³ volumes** (single-channel, brain-stripped, z-score normalised). Zongyu's data is 2-D 224×224 RGB Kaggle slices. The `adapt_2d_to_brainiac` helper in `brainiac_variants.py` (RGB→grayscale, trilinear inflate to 96³, per-volume z-score) is the pragmatic compromise — it produces inputs that are roughly in the distribution BrainIAC was trained on, but it's still feeding a "thin synthetic volume" to a model that learned from real 3-D MRI. For the genuinely informative Phase-4 comparison, use 3-D datasets (BraTS, UPenn-GBM, UCSF-PDGM) where BrainIAC's 3-D context is actually exploited.
2. **The two t-SNE embedding cells at the end of the fusion notebook are now patched** to use BrainIAC's CLS token directly (replacing the DenseNet `.features()` calls that don't exist on the ViT). If you see "BrainIAC embedding extract failed" warnings, the training and final test accuracy are unaffected — the visualization just falls back gracefully.
3. **BrainIAC's release license controls redistribution of the `.ckpt`.** Don't commit the weights file. Add `weights/` and `results/best_models/*.ckpt` to `.gitignore`.

## What stayed line-for-line identical

Everything that isn't a model-name reference:

- All CLI / config / data-loading cells in the notebook
- `seed_everything`, the random split, the worker_init_fn for reproducibility
- `CLIPTextEncoder` (ViT-B/16 with `laion2b_s34b_b88k`, multilingual prompt averaging, L2-normalised prototypes)
- `build_cache_from_dataset` (L2-normalised image features + one-hot labels)
- `TipAdapter` (the adapter MLP, the cosine similarity with `t_knn=0.07`, the alpha-blend)
- `OptimizedCLIPTipAdapter` (the wrapping module that takes raw images and returns the tuple)
- `SimpleFusionModel` (the sigmoid-gated fusion weight starting at 0.5, the `get_optimizer_params` layout)
- `train_simple_fusion_model` (the multi-task loss, AdamW with `weight_decay=0.01`, cosine schedule, val-best checkpointing, per-epoch test eval, embedding dump at the end)
- `BEST_BRAINIAC_CONFIG` (formerly `BEST_DENSENET_CONFIG`): same `backbone_lr=5e-5`, `head_lr=1e-3`, `focal_gamma=2.0`, `label_smoothing=0.05`
- `BEST_CLIP_CONFIG`: same `alpha=0.5`, `t_knn=0.07`, `lr_adapter=3e-4`

## Citation

```
Jia Y., Niu J., Li Z., Guo J. TumorCLIP: Lightweight Vision–Language Fusion
for Explainable MRI-Based Brain Tumor Classification. ISMRM 2026, Abstract 401-02-001.

Tak D., Garomsa B. A., Zapaishchykova A., et al. A generalizable foundation
model for analysis of human brain MRI. Nature Neuroscience 2026.
```
