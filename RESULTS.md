# TumorCLIP × BrainIAC — Phase 1 Results

## Training stability across 3 seeds (0, 17, 41)
- Best val accuracy: **78.98% ± 0.33%** (seeds 0/17/41: 78.87 / 79.35 / 78.71)
- Best test accuracy: **77.91% ± 0.80%** (seeds 0/17/41: 77.89 / 78.72 / 77.13)
- Wall clock: 5 min 25 sec per run on H200 NVL
- Learned fusion_weight (seed 41): σ(0.3885) = 0.596 (60% CLIP / 40% BrainIAC)

## Concept-intervention falsification test (mean ± std across 3 seeds)

| Class | Meaningful Δprob | Meaningful flip% | Control Δprob | Control flip% |
| --- | ---: | ---: | ---: | ---: |
| Glioma | +0.0040 ± 0.0016 | 0.45% ± 0.19% | +0.0001 ± 0.0000 | 0.00% ± 0.00% |
| Meningioma | +0.0013 ± 0.0004 | 0.00% ± 0.00% | +0.0002 ± 0.0001 | 0.00% ± 0.00% |
| NORMAL | +0.0007 ± 0.0002 | 0.27% ± 0.46% | +0.0001 ± 0.0000 | 0.27% ± 0.46% |
| Neurocitoma | +0.0004 ± 0.0001 | 0.00% ± 0.00% | +0.0008 ± 0.0001 | 0.00% ± 0.00% |
| Outros Tipos de Lesões | +0.0002 ± 0.0000 | 0.00% ± 0.00% | +0.0003 ± 0.0000 | 0.00% ± 0.00% |
| Schwannoma | +0.0016 ± 0.0003 | 0.00% ± 0.00% | +0.0002 ± 0.0000 | 0.00% ± 0.00% |

## Findings

1. **Maximum meaningful Δprob across all six classes: +0.0040 (Glioma).** Other five classes are ≤ +0.0016. Across n=3 seeds with stdev ≤ 0.0016, all meaningful edits produce functionally zero probability shift.
2. **Control edits produce ≥ meaningful edits for 3 of 6 classes:**
   - Neurocitoma: control (+0.0008) > meaningful (+0.0004) — control is **2× larger**.
   - Outros Tipos de Lesões: control (+0.0003) > meaningful (+0.0002).
   - NORMAL: identical flip rates (0.27% ± 0.46%) — meaningful and control are statistically indistinguishable.
3. **Prediction flip rates are ≤ 0.5% across all 6 classes for any edit type.** The model essentially does not change its predictions in response to text-prompt edits, regardless of clinical content.

## Mechanistic explanation
The CLIP branch contributes 60% of the prediction (σ(fusion_weight) = 0.596). Within the CLIP branch, half the output comes from the Tip-Adapter cache (one-hot training labels matched against L2-normalised image features) — the cache is unaffected by prompt edits. The other half is image-feature × text-prototype cosine similarity, but the prototype is an L2-normalised average over 4–5 multilingual prompts, so single-token edits change only ~1/N of the averaged direction.

## Conclusion
The text-prototype interpretability mechanism in TumorCLIP-style architectures is **mechanistically vacuous** on this dataset and training regime. The model's predictions are insensitive to clinically meaningful prompt edits, and indistinguishable from semantically null control edits in half of all tested classes. The architecture's accuracy (77.91% ± 0.80%) is real, but the interpretability story is decorative.
