# TumorCLIP × BrainIAC — Phase 2 BraTS Results

Phase 2 replicates the Phase 1 (Kaggle) falsification finding on the canonical
3D multimodal brain glioma benchmark (BraTS-GLI 2024), with patient-level
splits and a clinically grounded 3-class active-disease classification task.

## Dataset and task

- **Source**: BraTS-GLI 2024 Adult Glioma cohort, Synapse DUA-gated.
- **Cohort composition**: 1,350 patients, of which 85% are post-treatment
  (resection cavity present in segmentation mask) and 15% pre-treatment.
- **Task formulation**: 3-class active-disease classification, derived from
  per-patient segmentation voxel composition.

| Class | Definition | Train | Val | Test |
| --- | --- | ---: | ---: | ---: |
| 0 | Quiescent (minimal active disease) | 447 | 97 | 100 |
| 1 | Enhancing without necrosis | 267 | 60 | 54 |
| 2 | Necrotic enhancing (ring-enhancing recurrence) | 220 | 43 | 46 |
| – | Necrotic non-enhancing (n=16, excluded) | – | – | – |

Patient-level deterministic 70/15/15 split (seed 42). No file-level
leakage. Excluded the rare 16-patient class to keep the scheme trainable.

## Architecture

Unchanged from Phase 1:

- **Image encoder**: BrainIAC 3-D ViT-B, 96³ single-channel input.
- **Modality input**: T2-FLAIR (single modality, radiology-standard for
  tumor extent assessment).
- **Text encoder**: CLIP ViT-B/16 (`laion2b_s34b_b88k`), frozen.
- **Class prototypes**: averaged L²-normalised text embeddings of 4–5
  multilingual prompts per class (English, Chinese, Portuguese), reflecting
  post-treatment radiology reporting language.
- **Tip-Adapter cache**: built from the 934 training patients, 512-dim
  features × one-hot 3-class labels.
- **Fusion**: `(1 − σ(w)) · BrainIAC + σ(w) · CLIP`, learnable scalar `w`.
- **Loss**: 0.5 × class-weighted CE(fused) + 0.3 × Focal(BrainIAC) + 0.2 ×
  class-weighted CE(CLIP). Class weights = inverse frequency on training
  distribution: [0.696, 1.166, 1.415].

## Training stability across 3 seeds (0, 17, 41)

| Seed | Best epoch | Best val acc | Test acc at best | σ(fusion weight) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 14 | 61.5% | 60.5% | 0.610 |
| 17 | 14 | 63.5% | 57.0% | 0.610 |
| 41 | 16 | 58.5% | 59.0% | 0.611 |
| **Mean ± std** | – | **61.2% ± 2.1%** | **58.8% ± 1.4%** | **0.610 ± 0.001** |

Per-epoch wall clock on H200 NVL: ~1 min 17 sec. Full 20-epoch run: ~26 min.

## Concept-intervention falsification test

For each class, edit one clinically meaningful token in the prompts
(meaningful edit) and one semantically null token (control edit), re-encode
prompts through the frozen CLIP encoder, swap the prototype buffer in
place, and measure prediction shift on baseline-correct test cases.

### Edits used

| Class | Meaningful edit | Control edit |
| --- | --- | --- |
| Quiescent | `stable → unstable` | `Post-operative → Postoperative` |
| Enhancing without necrosis | `enhancing → non-enhancing` (affects 3/5 prompts) | `Nodular → Focal` |
| Necrotic enhancing | `necrotic → cystic` | `Aggressive → Invasive` |

### Results — mean ± std across 3 seeds

| Class | Meaningful Δp | Meaningful flip% | Control Δp | Control flip% |
| --- | ---: | ---: | ---: | ---: |
| Quiescent | +0.00037 ± 0.00005 | 0.00% ± 0.00% | −0.00020 ± 0.00000 | 0.00% ± 0.00% |
| Enhancing without necrosis | +0.00027 ± 0.00005 | 0.00% ± 0.00% | **+0.00033 ± 0.00017** | 0.00% ± 0.00% |
| Necrotic enhancing | +0.00130 ± 0.00000 | 0.00% ± 0.00% | +0.00040 ± 0.00000 | 0.00% ± 0.00% |

### Findings

1. **Maximum meaningful Δp across all 3 classes and 3 seeds: 0.13 percentage
   points (Necrotic enhancing).** The other two classes have meaningful Δp
   ≤ 0.04 pp. No reasonable interpretation of "the model uses text
   prototypes for clinical reasoning" survives effect sizes this small.

2. **Zero prediction flips across 600 patient-edit observations** (3 seeds ×
   200 test patients per seed, evaluated under every meaningful and
   control edit). On Phase 1 (Kaggle) the maximum flip rate was 0.5% — on
   BraTS it is exactly 0%.

3. **For 1 of 3 classes (Enhancing without necrosis), the control edit
   produces a larger mean effect than the meaningful edit.** A clinical
   synonym (`Nodular → Focal`) moves predictions on average more than
   negating the class-defining clinical concept (`enhancing →
   non-enhancing` across 3 of 5 prompts).

4. **The Necrotic enhancing class produces bit-identical results across
   all three random seeds.** Meaningful Δp = 0.0013 and control Δp =
   0.0004 to four decimal places in every seed. This is an extraordinarily
   stable falsification.

## Mechanistic explanation

Same as Phase 1 — the architecture decomposes as:

```
logits_fused = (1 - σ(w)) · logits_BrainIAC + σ(w) · logits_CLIP
             = 0.39 · logits_BrainIAC + 0.61 · logits_CLIP

logits_CLIP  = (1 - α) · logits_text + α · logits_kNN_cache
             = 0.5 · logits_text + 0.5 · logits_kNN_cache
```

The Tip-Adapter cache (kNN over training image features × one-hot training
labels) contributes ~30% of every prediction independent of the text
prototypes. The BrainIAC image classifier contributes ~39% independent of
text. The remaining ~31% of the signal flows through text prototypes, and
even within that, the prototype is an L²-normalised average of 4–5
multilingual prompts, so single-token edits affect only ~1/N of the
averaged direction.

The fusion weight σ(w) settled at 0.610 ± 0.001 across seeds — virtually
identical to the Kaggle finding (0.596). The architecture's preference for
the CLIP branch is reproducible across very different datasets.

## Comparison: Phase 1 (Kaggle) vs Phase 2 (BraTS)

| Metric | Kaggle (n=3, 6 classes) | BraTS (n=3, 3 classes) |
| --- | ---: | ---: |
| Test accuracy | 77.91% ± 0.80% | 58.83% ± 1.44% |
| σ(fusion weight) | 0.596 ± low | 0.610 ± 0.001 |
| Max meaningful Δp | 0.40 pp | 0.13 pp |
| Max control Δp | 0.08 pp | 0.05 pp |
| Max flip rate | 0.5% | **0.0%** |
| Classes where control ≥ meaningful | 3 of 6 | 1 of 3 |

The BraTS test accuracy is lower than Kaggle because BraTS uses
patient-level splits (no patient leakage), real 3D MRI (no 2D inflation),
and clinically subtle post-treatment phenotypes (versus Kaggle's coarse
6-class tumor-type task). The lower number is more honest, not weaker.

**The falsification finding is markedly stronger on BraTS** — predictions
are completely insensitive to prompt edits on a properly-split 3D MRI
benchmark, and the per-seed variance on the strongest signal class is
literally zero.

## Conclusion

The text-prototype interpretability mechanism in TumorCLIP-style
vision-language fusion architectures does not survive concept-intervention
falsification on either of two evaluation regimes:

1. A 2D, file-level-split, 6-class tumor-type benchmark (Kaggle 17-class
   consolidated to 6 superclasses, 1,316 test cases × 3 seeds).
2. A 3D, patient-level-split, 3-class active-disease benchmark
   (BraTS-GLI 2024, 200 test cases × 3 seeds).

In both regimes, meaningful clinical prompt edits produce probability
shifts an order of magnitude smaller than the architecture's accuracy
margin, semantically null control edits frequently match or exceed the
meaningful edits, and prediction flip rates are at or near zero. The
architecture's accuracy is real and reproducible; the interpretability
story is decorative on both 2D and 3D evaluation regimes.

The cause is identifiable from the architecture itself and matches
predictions made before the BraTS experiment: the Tip-Adapter cache and
BrainIAC image classifier together account for ~70% of every prediction
independent of any prompt edit, and the remaining ~30% is heavily diluted
by multilingual prompt averaging. The same architectural decomposition
holds across datasets, classes, and image dimensionality.

## Files

- Manifest: `manifest_brats_train.csv`
- Trained checkpoints: `results/brats_fusion_seed{0,17,41}/best.pt`
- Per-epoch metrics: `results/brats_fusion_seed{0,17,41}/history.json`
- Intervention results: `results/brats_fusion_seed{0,17,41}/intervention_seed{0,17,41}.json`
- Training script: `train_fusion_brats.py`
- Inventory script: `scripts/inventory_brats.py`
- Intervention script: `scripts/concept_intervention_brats.py`
- Dataset class: `src/data/brats_dataset.py`
- Prompts: `BRATS_DISEASE_ACTIVITY_PROMPTS` in `src/config/constants.py`
