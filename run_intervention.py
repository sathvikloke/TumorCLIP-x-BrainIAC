import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.constants import CLASS_NAMES, NUM_CLASSES
from src.models.fusion_components import SimpleFusionModel, create_clip_brainiac_model
from scripts.concept_intervention import run_intervention
from train_fusion import build_data_loaders


def main():
    parser = argparse.ArgumentParser(description="Concept-intervention falsification test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--brainiac_weights", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--class_name", default="Glioma", choices=CLASS_NAMES)
    parser.add_argument("--find", default="Ring-enhancing")
    parser.add_argument("--replace", default="Non-enhancing")
    parser.add_argument("--control_find", default="showing")
    parser.add_argument("--control_replace", default="displaying")
    parser.add_argument("--out", default="results/concept_intervention.json")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.brainiac_weights:
        os.environ["BRAINIAC_WEIGHTS_PATH"] = args.brainiac_weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("\n=== Building data loaders ===")
    train_loader, val_loader, test_loader = build_data_loaders(
        args.data_root, args.batch_size, seed=41, num_workers=args.num_workers,
    )
    print(f"  train: {len(train_loader.dataset)}, test: {len(test_loader.dataset)}")

    print("\n=== Rebuilding fusion model architecture ===")
    image_encoder, text_encoder = create_clip_brainiac_model(embed_dim=512, dropout=0.1)
    if args.brainiac_weights and os.path.exists(args.brainiac_weights):
        image_encoder.load_pretrained_weights(args.brainiac_weights)

    BEST_BRAINIAC_CONFIG = {"backbone_lr": 5e-5, "head_lr": 1e-3,
                            "focal_gamma": 2.0, "label_smoothing": 0.05}
    BEST_CLIP_CONFIG = {"alpha": 0.5, "t_knn": 0.07, "lr_adapter": 3e-4}

    model = SimpleFusionModel(
        brainiac_config=BEST_BRAINIAC_CONFIG,
        clip_config=BEST_CLIP_CONFIG,
        num_classes=NUM_CLASSES,
    ).to(device)
    if args.brainiac_weights and os.path.exists(args.brainiac_weights):
        model.load_brainiac_weights(args.brainiac_weights)

    image_encoder = image_encoder.to(device)
    model.share_backbone_with(image_encoder)
    model.setup_clip_branch(train_loader, device, image_encoder, text_encoder)

    print(f"\n=== Loading trained checkpoint: {args.checkpoint} ===")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  {len(missing)} missing keys (first 3: {missing[:3]})")
    if unexpected:
        print(f"  {len(unexpected)} unexpected keys (first 3: {unexpected[:3]})")
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")

    print(f"\n=== Running concept intervention ===")
    print(f"  Meaningful edit: '{args.find}' -> '{args.replace}'")
    if args.control_find:
        print(f"  Control edit:    '{args.control_find}' -> '{args.control_replace}'")

    run_intervention(
        model=model,
        text_encoder=text_encoder,
        loader=test_loader,
        device=device,
        class_name=args.class_name,
        find=args.find,
        replace=args.replace,
        control_find=args.control_find,
        control_replace=args.control_replace,
        save_to=args.out,
    )


if __name__ == "__main__":
    main()
