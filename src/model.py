"""
TumorCLIP model.

Architecture (per the ISMRM 2026 abstract):

    image -> [Backbone] -> image_features (D-dim)
                              |
                              +-> [image_classifier] -> image_logits (C)
                              |
                              +-> [projection MLP] -> projected (E-dim)
                                       |
                                       +-> cosine_sim(text_prototypes) -> text_logits (C)

    final_logits = (1 - alpha) * image_logits + alpha * text_logits

The Backbone class wraps any image encoder that returns a (B, D) feature tensor.
For Phase 1, it's a timm DenseNet121. For Phase 2, replace with a BrainIAC
wrapper that returns a (B, 2048) feature tensor from 3D volumes — the head
itself does not change.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimmBackbone(nn.Module):
    """Wraps any timm model as a feature extractor returning (B, D)."""

    def __init__(
        self,
        name: str = "densenet121",
        pretrained: bool = True,
        freeze: bool = False,
    ):
        super().__init__()
        # num_classes=0 + global_pool='avg' returns pooled features instead of logits
        self.backbone = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.feature_dim = self.backbone.num_features
        self.frozen = freeze

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True) -> "TimmBackbone":
        """Override train() so a frozen backbone stays in eval mode.

        Without this, every model.train() call at the start of an epoch flips
        the backbone — including its BatchNorm layers — back into train mode,
        which leaks the new dataset's statistics into the frozen running stats.
        """
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, D)


class ProjectionMLP(nn.Module):
    """Small 2-layer projector from backbone feature dim to text-embedding dim."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TumorCLIPHead(nn.Module):
    """The TumorCLIP fusion head — takes pooled image features + frozen text prototypes.

    This is the part you reuse across backbones in Phase 2.

    Args:
        in_features: dimension of the backbone's pooled feature.
        n_classes: number of output classes.
        text_prototypes: tensor of shape (n_classes, embed_dim), L2-normalized.
        alpha: fusion weight (paper-optimal: 0.3).
        projection_hidden: hidden dim of the projection MLP.
        learnable_alpha: if True, alpha is a trainable parameter.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        text_prototypes: torch.Tensor,
        alpha: float = 0.3,
        projection_hidden: int = 512,
        learnable_alpha: bool = False,
    ):
        super().__init__()
        assert text_prototypes.shape[0] == n_classes, (
            f"Expected {n_classes} text prototypes, got {text_prototypes.shape[0]}."
        )

        embed_dim = text_prototypes.shape[1]
        self.image_classifier = nn.Linear(in_features, n_classes)
        self.projection = ProjectionMLP(in_features, projection_hidden, embed_dim)

        # Register prototypes as a buffer so they move with .to(device) but
        # are not trainable. Clone defensively so later mutation of the input
        # tensor by the caller can't silently change the model.
        self.register_buffer("text_prototypes", text_prototypes.detach().clone())

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))

        # Temperature for cosine-similarity logits — borrowed from CLIP.
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # exp = ~14.3

    def forward(self, image_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns a dict with image_logits, text_logits, and fused logits."""
        image_logits = self.image_classifier(image_features)

        projected = self.projection(image_features)
        projected = F.normalize(projected, dim=-1)
        # text_prototypes are pre-normalized; logit_scale (~14.3) plays CLIP's
        # temperature role so the cosine similarities behave as logits.
        text_logits = self.logit_scale.exp() * (projected @ self.text_prototypes.t())

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        logits = (1.0 - alpha) * image_logits + alpha * text_logits

        return {
            "logits": logits,
            "image_logits": image_logits,
            "text_logits": text_logits,
        }


class TumorCLIP(nn.Module):
    """End-to-end TumorCLIP: backbone + head, wired together."""

    def __init__(
        self,
        backbone: nn.Module,
        text_prototypes: torch.Tensor,
        n_classes: int = 6,
        alpha: float = 0.3,
        projection_hidden: int = 512,
        learnable_alpha: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        # Try to infer feature_dim from the backbone, fall back to a manual probe.
        # When probing, place the dummy tensor on whatever device the backbone is on.
        feature_dim = getattr(backbone, "feature_dim", None)
        if feature_dim is None:
            try:
                probe_device = next(backbone.parameters()).device
            except StopIteration:
                probe_device = torch.device("cpu")
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=probe_device)
                feature_dim = backbone(dummy).shape[-1]

        self.head = TumorCLIPHead(
            in_features=feature_dim,
            n_classes=n_classes,
            text_prototypes=text_prototypes,
            alpha=alpha,
            projection_hidden=projection_hidden,
            learnable_alpha=learnable_alpha,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return self.head(features)


def _smoke_test() -> None:
    """Run a synthetic forward pass with random text prototypes.

    Use:
        python -m src.model
    """
    import torch
    n_classes = 6
    embed_dim = 512
    fake_protos = torch.randn(n_classes, embed_dim)
    fake_protos = fake_protos / fake_protos.norm(dim=-1, keepdim=True)

    backbone = TimmBackbone("densenet121", pretrained=False)
    model = TumorCLIP(backbone, fake_protos, n_classes=n_classes)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("forward pass OK")
    print(f"  logits      : {tuple(out['logits'].shape)} (expect (2, {n_classes}))")
    print(f"  image_logits: {tuple(out['image_logits'].shape)}")
    print(f"  text_logits : {tuple(out['text_logits'].shape)}")
    assert out["logits"].shape == (2, n_classes)
    assert out["image_logits"].shape == (2, n_classes)
    assert out["text_logits"].shape == (2, n_classes)
    print("shape assertions OK")


if __name__ == "__main__":
    _smoke_test()
