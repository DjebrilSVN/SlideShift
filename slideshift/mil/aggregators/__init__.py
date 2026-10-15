"""MIL Aggregator registry and exports.

Available aggregators
---------------------
Baseline:
  mean_pool   — Mean pooling (no learned weights)
  max_pool    — Max pooling

Attention-based:
  abmil       — Attention MIL (Ilse et al., 2018) — gated dual-stream
  clam_sb     — CLAM single-branch (Lu et al., 2021)
  clam_mb     — CLAM multi-branch

Transformer:
  transmil    — TransMIL (Shao et al., 2021) with CLS token

Graph:
  patch_gcn   — Patch-GCN spatial k-NN graph (Chen et al., 2021)

Double-tier:
  dtfd_mil    — DTFD-MIL (Zhang et al., CVPR 2022) pseudo-bag distillation

Adaptive multi-head:
  acmil       — ACMIL multi-branch adaptive aggregation (Yang et al., 2023)
"""

from typing import Any, Dict

from slideshift.mil.aggregators.pooling import MeanPoolingMIL, MaxPoolingMIL
from slideshift.mil.aggregators.abmil import AttentionMIL
from slideshift.mil.aggregators.clam import CLAM_SB, CLAM_MB
from slideshift.mil.aggregators.transmil import TransMIL
from slideshift.mil.aggregators.patch_gcn import PatchGCN
from slideshift.mil.aggregators.dtfd_mil import DTFDMIL
from slideshift.mil.aggregators.acmil import ACMIL


AGGREGATOR_REGISTRY: Dict[str, Any] = {
    "mean_pool": MeanPoolingMIL,
    "max_pool": MaxPoolingMIL,
    "abmil": AttentionMIL,
    "clam_sb": CLAM_SB,
    "clam_mb": CLAM_MB,
    "transmil": TransMIL,
    "patch_gcn": PatchGCN,
    "dtfd_mil": DTFDMIL,
    "acmil": ACMIL,
}


def build_aggregator(
    architecture: str,
    input_dim: int,
    num_classes: int = 2,
    hidden_dim: int = 256,
    dropout: float = 0.25,
    **kwargs: Any,
):
    """Instantiate a Multiple-Instance Learning aggregator model.

    Args:
        architecture: One of the keys in AGGREGATOR_REGISTRY.
        input_dim:    Tile embedding dimension (from the foundation model).
        num_classes:  Number of output classes.
        hidden_dim:   Hidden layer size.
        dropout:      Dropout probability.
        **kwargs:     Extra kwargs forwarded to the aggregator constructor.

    Returns:
        Instantiated aggregator object.
    """
    arch_clean = architecture.lower().strip()
    if arch_clean not in AGGREGATOR_REGISTRY:
        raise ValueError(
            f"Unknown MIL architecture '{architecture}'. "
            f"Options: {list(AGGREGATOR_REGISTRY.keys())}"
        )

    cls_obj = AGGREGATOR_REGISTRY[arch_clean]

    # Pooling baselines take a simpler signature
    if arch_clean in ("mean_pool", "max_pool"):
        return cls_obj(input_dim=input_dim, num_classes=num_classes, dropout=dropout)

    # All learned aggregators share (input_dim, hidden_dim, num_classes, dropout)
    return cls_obj(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs,
    )


__all__ = [
    "AGGREGATOR_REGISTRY",
    "build_aggregator",
    "MeanPoolingMIL",
    "MaxPoolingMIL",
    "AttentionMIL",
    "CLAM_SB",
    "CLAM_MB",
    "TransMIL",
    "PatchGCN",
    "DTFDMIL",
    "ACMIL",
]
