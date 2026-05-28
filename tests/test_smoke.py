"""
Self-contained smoke test for the TumorCLIP reimplementation.

Run after `pip install -r requirements.txt`:

    python -m tests.test_smoke

It exercises, in order:

  1. Text-prototype encoding via the frozen CLIP text encoder.
  2. TumorCLIP forward pass with the real DenseNet121 backbone (untrained).
  3. The KaggleTumorDataset over a synthetic 6-class directory.
  4. The evaluate() metric pipeline through a real DataLoader.
  5. The concept_intervention() falsification protocol.

Each step prints PASS / FAIL. Any FAIL aborts the run with a traceback.

Tested with: torch>=2.0, timm>=0.9, open_clip_torch>=2.20.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

# Allow running as `python -m tests.test_smoke` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import KaggleTumorDataset
from src.evaluate import concept_intervention, evaluate
from src.model import TimmBackbone, TumorCLIP
from src.prototypes import CLASS_NAMES, build_prototype_bank


def _make_synthetic_dataset(root: Path, n_per_class: int = 4) -> None:
    """Create a fake 6-class dataset of small RGB images for both train and test."""
    for split in ("train", "test"):
        for cls in CLASS_NAMES:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                Image.new("RGB", (32, 32), color=(i * 30, 100, 200)).save(d / f"img_{i}.jpg")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 1. Prototype bank
    print("\n[1/5] building text prototype bank ...")
    bank = build_prototype_bank(device=device)
    assert bank.embeddings.shape == (6, 512), f"bad shape: {bank.embeddings.shape}"
    assert torch.allclose(bank.embeddings.norm(dim=-1), torch.ones(6), atol=1e-4)
    print(f"     prototype shape {tuple(bank.embeddings.shape)}, L2-normalized: PASS")

    # 2. Model forward
    print("\n[2/5] building TumorCLIP and running a forward pass ...")
    backbone = TimmBackbone("densenet121", pretrained=False)
    model = TumorCLIP(backbone, bank.embeddings, n_classes=6).to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out["logits"].shape == (2, 6)
    assert out["image_logits"].shape == (2, 6)
    assert out["text_logits"].shape == (2, 6)
    print("     forward pass shapes (2, 6) x 3: PASS")

    # 3. Dataset
    print("\n[3/5] building synthetic dataset and verifying loader ...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "data"
        _make_synthetic_dataset(tmp_path, n_per_class=4)
        train_ds = KaggleTumorDataset(tmp_path, "train", image_size=224)
        test_ds = KaggleTumorDataset(tmp_path, "test", image_size=224)
        assert len(train_ds) == 6 * 4
        assert len(test_ds) == 6 * 4
        img, lbl = train_ds[0]
        assert img.shape == (3, 224, 224)
        assert 0 <= lbl < 6
        print(f"     train={len(train_ds)} test={len(test_ds)} item-shape={tuple(img.shape)}: PASS")

        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

        # 4. Evaluate
        print("\n[4/5] running evaluate() through the model ...")
        result = evaluate(model, test_loader, device)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.macro_f1 <= 1.0
        assert result.confusion.shape == (6, 6)
        assert len(result.per_class_recall) == 6
        print(f"     acc={result.accuracy:.3f} macroF1={result.macro_f1:.3f}: PASS")

        # 5. Concept intervention — pick a class the model happened to predict
        #    correctly at least once. Since the model is untrained, this is
        #    rare; fall back to forcing predictions by hot-wiring image_logits.
        print("\n[5/5] running concept_intervention() ...")
        # Force at least one true positive for Meningioma by editing the bias
        # of the image_classifier so the model always picks Meningioma. This
        # is just to exercise the code path — interpretability claims rely on
        # a *trained* model.
        with torch.no_grad():
            model.head.image_classifier.bias.zero_()
            model.head.image_classifier.bias[CLASS_NAMES.index("Meningioma")] = 100.0
        try:
            interv = concept_intervention(
                model,
                test_loader,
                device,
                class_name="Meningioma",
                find="dural tail",
                replace="cortical band",
            )
            assert interv.n_positives > 0
            print(
                f"     n_pos={interv.n_positives} "
                f"mean_drop={interv.mean_prob_drop:.4f} "
                f"flipped={interv.fraction_flipped:.2f}: PASS"
            )
        except Exception as exc:
            print(f"     concept_intervention FAILED: {exc!r}")
            raise

    print("\nAll 5 smoke tests passed.")


if __name__ == "__main__":
    main()
