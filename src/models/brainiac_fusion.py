"""Deprecated. The fusion model now comes straight from Zongyu's notebook
(``BrainIAC_CLIP_Fusion_Model_Training.ipynb``) with DenseNet swapped for
BrainIAC. There is no separate Python module for it on purpose — the
fusion architecture is defined inline in the notebook, exactly as in the
original TumorCLIP release.
"""
raise ImportError(
    "src.models.brainiac_fusion is deprecated. "
    "Run BrainIAC_CLIP_Fusion_Model_Training.ipynb instead."
)
