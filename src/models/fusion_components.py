"""Fusion components for the BrainIAC × CLIP TumorCLIP pipeline.

This module holds the classes that used to be defined inline in Zongyu's
notebook cells (CLIPTextEncoder, TipAdapter, OptimizedCLIPTipAdapter,
SimpleFusionModel) and a helper to build the Tip-Adapter cache from a
training loader.

The logic is preserved exactly from Zongyu's CLIP_Fusion_Model_Training
notebook (with DenseNet → BrainIAC substitutions). Only the indentation
is normalized to standard 4-space Python.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.config.constants import CLASS_NAMES, NUM_CLASSES, PROFESSIONAL_MEDICAL_PROMPTS
from src.models.brainiac_variants import BrainIACClassifier, BrainIACEncoder
from src.models.losses import FocalLoss, LabelSmoothingCrossEntropy


# ---------------------------------------------------------------------------
# CLIP text encoder
# ---------------------------------------------------------------------------

class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder that builds multilingual prototypes."""

    def __init__(self, clip_model="ViT-B-16", pretrained="laion2b_s34b_b88k", class_names=None):
        super().__init__()
        import open_clip

        self.class_names = class_names or CLASS_NAMES

        available_models = [pretrained, "laion2b_s34b_b88k", "openai", "laion400m_e32"]
        loaded = False
        for model_name in available_models:
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    clip_model, pretrained=model_name
                )
                self.tokenizer = open_clip.get_tokenizer(clip_model)
                print(f"Successfully loaded CLIP model: {clip_model} ({model_name})")
                loaded = True
                break
            except Exception as e:
                print(f"Warning: Failed to load {model_name}: {e}")
                continue
        if not loaded:
            raise RuntimeError("Unable to load any CLIP pre-trained model")

        # Freeze the text encoder
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Get CLIP text feature dim
        if hasattr(self.model, "text_projection"):
            self.clip_dim = self.model.text_projection.shape[1]
        else:
            self.clip_dim = self.model.ln_final.weight.shape[0]

        print(f"CLIP text encoder: {len(self.class_names)} classes, {self.clip_dim} dim")

    @torch.no_grad()
    def encode_prompts(self, prompts):
        tokens = self.tokenizer(prompts).to(next(self.model.parameters()).device)
        text_features = self.model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def build_text_prototypes(self, prompts_dict=None, device=None):
        device = device or next(self.model.parameters()).device
        prompts_dict = prompts_dict or PROFESSIONAL_MEDICAL_PROMPTS

        prototypes = []
        for class_name in self.class_names:
            class_prompts = prompts_dict.get(
                class_name, [f"MRI of a brain with {class_name.lower()}"]
            )
            text_features = self.encode_prompts(class_prompts)
            prototype = text_features.mean(dim=0, keepdim=True)
            prototype = F.normalize(prototype, dim=-1)
            prototypes.append(prototype)
        return torch.cat(prototypes, dim=0).to(device)

    def forward(self, prompts_dict=None):
        return self.build_text_prototypes(prompts_dict)


def create_clip_brainiac_model(embed_dim=512, dropout=0.1):
    """Construct the (image_encoder, text_encoder) pair used by the fusion model."""
    print("Creating CLIP + BrainIAC model...")
    image_encoder = BrainIACEncoder(embed_dim=embed_dim, dropout=dropout)
    text_encoder = CLIPTextEncoder()
    return image_encoder, text_encoder


# ---------------------------------------------------------------------------
# Tip-Adapter cache + adapter modules
# ---------------------------------------------------------------------------

def build_cache_from_dataset(image_encoder, text_encoder, train_loader, device="cuda"):
    """Build the Tip-Adapter cache (keys, values) from the training set."""
    print("Building Tip-Adapter cache from training set...")
    if train_loader is None:
        raise ValueError("train_loader is None - cannot build cache from empty dataset")

    image_encoder = image_encoder.to(device).eval()
    text_encoder = text_encoder.to(device).eval()

    cache_keys = []
    cache_values = []
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Building cache"):
            images = images.to(device)
            labels = labels.to(device)
            image_features = image_encoder.forward_features(images)
            image_features = F.normalize(image_features, dim=-1)
            one_hot = F.one_hot(labels, num_classes=NUM_CLASSES).float()
            cache_keys.append(image_features.cpu())
            cache_values.append(one_hot.cpu())
    cache_keys = torch.cat(cache_keys, dim=0)
    cache_values = torch.cat(cache_values, dim=0)
    print(f"Cache built: {cache_keys.shape[0]} samples")
    return cache_keys, cache_values


class TipAdapter(nn.Module):
    """Tip-Adapter with a trainable 2-layer adapter MLP."""

    def __init__(self, clip_model, cache_keys, cache_values, alpha=0.5, t_knn=0.07,
                 lr_adapter=3e-4, enable_training=True):
        super().__init__()
        self.clip_model = clip_model  # text prototypes tensor
        self.alpha = alpha
        self.t_knn = t_knn
        self.lr_adapter = lr_adapter

        self.register_buffer("cache_keys", cache_keys)
        self.register_buffer("cache_values", cache_values)

        if enable_training:
            self.adapter = nn.Sequential(
                nn.Linear(cache_keys.shape[1], cache_keys.shape[1] // 4),
                nn.ReLU(),
                nn.Linear(cache_keys.shape[1] // 4, cache_keys.shape[1]),
            )
        else:
            self.adapter = None

    def forward(self, image_features):
        image_features = F.normalize(image_features, dim=-1)
        if self.adapter is not None:
            adapted_features = self.adapter(image_features)
            adapted_features = F.normalize(adapted_features, dim=-1)
        else:
            adapted_features = image_features

        cache_keys_norm = F.normalize(self.cache_keys, dim=-1)
        similarities = torch.mm(adapted_features, cache_keys_norm.t())
        similarities = similarities / self.t_knn
        weights = F.softmax(similarities, dim=-1)
        knn_logits = torch.mm(weights, self.cache_values)

        clip_logits = torch.mm(image_features, self.clip_model.t())
        final_logits = (1 - self.alpha) * clip_logits + self.alpha * knn_logits
        return final_logits, knn_logits, clip_logits

    def get_adapter_params(self):
        if self.adapter is not None:
            return list(self.adapter.parameters())
        return []


class OptimizedCLIPTipAdapter(nn.Module):
    """Wraps the image encoder + text prototypes + TipAdapter together."""

    def __init__(self, image_encoder, text_encoder, cache_keys, cache_values,
                 alpha=0.5, t_knn=0.07, lr_adapter=3e-4, device="cuda"):
        super().__init__()
        self.image_encoder = image_encoder.to(device)
        self.text_encoder = text_encoder.to(device)
        self.device = device

        with torch.no_grad():
            text_prototypes = self.text_encoder().to(device)
        self.register_buffer("text_prototypes", text_prototypes)

        cache_keys = cache_keys.to(device)
        cache_values = cache_values.to(device)

        self.tip_adapter = TipAdapter(
            clip_model=self.text_prototypes,
            cache_keys=cache_keys,
            cache_values=cache_values,
            alpha=alpha,
            t_knn=t_knn,
            lr_adapter=lr_adapter,
        ).to(device)

    def forward(self, images, mode="eval"):
        image_features = self.image_encoder.forward_features(images)
        final_logits, knn_logits, clip_logits = self.tip_adapter(image_features)
        if mode == "eval":
            return final_logits
        return final_logits, knn_logits, clip_logits

    def get_trainable_params(self):
        params = []
        params.extend(self.tip_adapter.get_adapter_params())
        params.extend(list(self.image_encoder.feature_projection.parameters()))
        return params


# ---------------------------------------------------------------------------
# SimpleFusionModel (BrainIAC + CLIP)
# ---------------------------------------------------------------------------

class SimpleFusionModel(nn.Module):
    """Simplified fusion model combining BrainIAC and CLIP."""

    def __init__(self, brainiac_config, clip_config, num_classes=6):
        super().__init__()
        print("Creating simplified fusion model")

        self.brainiac_branch = BrainIACClassifier(
            num_classes=num_classes,
            backbone_lr=brainiac_config["backbone_lr"],
            head_lr=brainiac_config["head_lr"],
            focal_gamma=brainiac_config["focal_gamma"],
            label_smoothing=brainiac_config["label_smoothing"],
        )

        self.clip_config = clip_config
        self.clip_branch = None
        self.device = None

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        print("Simplified fusion model created")

    def share_backbone_with(self, image_encoder):
        """Replace the BrainIAC branch's backbone with the encoder's backbone.

        Halves GPU memory when both branches use the same ViT-B instance.
        Call this after construction, before training begins.
        """
        try:
            self.brainiac_branch.backbone = image_encoder.backbone
            print("Sharing BrainIAC backbone between fusion branches.")
            return True
        except Exception as e:
            print(f"Could not share backbones ({e}); using independent copies.")
            return False

    def setup_clip_branch(self, train_loader, device, image_encoder=None, text_encoder=None):
        """Build the Tip-Adapter cache and attach the CLIP branch."""
        print("Setting up CLIP branch")
        if image_encoder is None or text_encoder is None:
            print("No available encoders found, creating...")
            image_encoder, text_encoder = create_clip_brainiac_model()

        image_encoder = image_encoder.to(device)
        text_encoder = text_encoder.to(device)

        cache_keys, cache_values = build_cache_from_dataset(
            image_encoder, text_encoder, train_loader, device
        )

        self.clip_branch = OptimizedCLIPTipAdapter(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            cache_keys=cache_keys,
            cache_values=cache_values,
            alpha=self.clip_config["alpha"],
            t_knn=self.clip_config["t_knn"],
            lr_adapter=self.clip_config["lr_adapter"],
        ).to(device)

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.device = device
        print(f"CLIP branch setup complete, cache shape: {cache_keys.shape}")
        return True

    def load_brainiac_weights(self, checkpoint_path):
        return self.brainiac_branch.load_pretrained_weights(checkpoint_path)

    def forward(self, images, mode="eval"):
        brainiac_logits = self.brainiac_branch(images)
        if self.clip_branch is None:
            return brainiac_logits
        try:
            if mode == "train":
                clip_final, _, _ = self.clip_branch(images, mode="train")
            else:
                clip_final = self.clip_branch(images, mode="eval")
            fusion_weight = torch.sigmoid(self.fusion_weight)
            fused_logits = (1 - fusion_weight) * brainiac_logits + fusion_weight * clip_final
            return fused_logits, brainiac_logits, clip_final
        except Exception as e:
            print(f"CLIP branch execution failed, using BrainIAC: {str(e)}")
            return brainiac_logits

    def get_optimizer_params(self):
        params = []
        params.extend(self.brainiac_branch.get_optimizer_params())
        if self.clip_branch is not None:
            try:
                clip_params = self.clip_branch.get_trainable_params()
                params.append({
                    "params": clip_params,
                    "lr": self.clip_config["lr_adapter"],
                    "name": "clip",
                })
            except Exception:
                pass
        params.append({
            "params": [self.fusion_weight],
            "lr": 1e-4,
            "name": "fusion",
        })
        return params
