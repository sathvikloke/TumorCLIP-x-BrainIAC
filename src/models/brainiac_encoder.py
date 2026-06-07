"""Deprecated. The line-for-line copy of Zongyu's pipeline now lives in
``src/models/brainiac_variants.py`` (mirroring ``densenet_variants.py``).
Use ``BrainIACClassifier`` / ``BrainIACEncoder`` from there.
"""
raise ImportError(
    "src.models.brainiac_encoder is deprecated. "
    "Use src.models.brainiac_variants (BrainIACClassifier, BrainIACEncoder) instead."
)
