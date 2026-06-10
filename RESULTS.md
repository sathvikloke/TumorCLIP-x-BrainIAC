# TumorCLIP × BrainIAC — Phase 1 Results

## Training (seed 41, fusion_run1)
- Backbone: BrainIAC 3-D ViT-B (96³ input, CLS token)
- Data: Kaggle 17-class consolidated to 6 superclasses (2,479 train / 620 val / 1,316 test)
- Wall clock on H200: 5 min 25 sec for 15 epochs
- Best val accuracy: 78.71% (epoch 12)
- Best test accuracy: 77.13%
- Learned fusion_weight: σ(0.3885) = 0.596 (60% CLIP / 40% BrainIAC)

## Concept-intervention sweep across all 6 classes

| Class | Meaningful edit | Δ prob | Flipped | Control edit | Δ prob | Flipped |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Glioma | glioma→schwannoma (4/5) | +0.0059 | 0.67% | showing→displaying | +0.0001 | 0.00% |
| Meningioma | meningioma→glioma (2/5) | +0.0017 | 0.00% | consistent→associated | +0.0002 | 0.00% |
| NORMAL | normal→abnormal (3/5) | +0.0008 | 0.80% | brain→cerebral | +0.0001 | 0.80% |
| Neurocitoma | neurocytoma→schwannoma (2/4) | +0.0005 | 0.00% | Intraventricular→Extra-axial | +0.0009 | 0.00% |
| Outros Tipos de Lesões | lesions→tumors (1/4) | +0.0002 | 0.00% | intracranial→external | +0.0003 | 0.00% |
| Schwannoma | schwannoma→glioma (2/4) | +0.0016 | 0.00% | Extra-axial→Intra-axial | +0.0002 | 0.00% |

## Interpretation
Across all 6 classes, meaningful edits to text prototypes produce ≤ 0.6 percentage points of probability shift, and ≤ 0.8% prediction flips. Control edits produce comparable, sometimes larger, shifts (Neurocitoma). The interpretability claim that text prototype edits produce traceable changes in predictions is not supported by these results.

Mechanistic explanation: the Tip-Adapter cache contributes 50% of the CLIP branch output (alpha=0.5). The cache stores one-hot training labels matched against L2-normalised image features and is unaffected by prompt edits. Even within the text-similarity half of the CLIP branch, the prototype is an L2-normalised average of 4-5 multilingual prompts, so single-token edits affect only 1/N of the averaged direction.
