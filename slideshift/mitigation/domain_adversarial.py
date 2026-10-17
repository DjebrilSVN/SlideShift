"""Domain-Adversarial Neural Network (DANN) for site-invariant slide representations."""

from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.autograd import Function
    HAS_TORCH = hasattr(torch, "__version__") and hasattr(nn, "Module")
except (ImportError, AttributeError):
    torch = None
    nn = None
    Function = object
    HAS_TORCH = False


if HAS_TORCH:
    class GradientReversalFunction(Function):
        """Gradient Reversal Layer (GRL) from Ganin et al., 2016."""

        @staticmethod
        def forward(ctx, x, alpha: float):
            ctx.alpha = alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.alpha, None


    def grad_reverse(x: "torch.Tensor", alpha: float = 1.0) -> "torch.Tensor":
        return GradientReversalFunction.apply(x, alpha)


    class DomainAdversarialMIL(nn.Module):
        """MIL classifier augmented with a domain adversary head to enforce site-invariance."""

        def __init__(
            self,
            base_mil_model: nn.Module,
            hidden_dim: int = 256,
            num_sites: int = 5,
            adversary_weight: float = 0.10,
        ):
            super().__init__()
            self.base_mil = base_mil_model
            self.num_sites = num_sites
            self.adversary_weight = adversary_weight

            # Adversary discriminator predicting hospital/scanner site
            self.site_discriminator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_sites),
            )

        def forward(
            self,
            h: "torch.Tensor",
            alpha: float = 1.0,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Returns:
                (class_logits, attention_weights, slide_embedding, site_logits)
            """
            out = self.base_mil(h)
            class_logits = out[0]
            attention_weights = out[1]
            slide_embedding = out[2]

            # Invert gradient through slide embedding to domain adversary
            reversed_emb = grad_reverse(slide_embedding, alpha)
            site_logits = self.site_discriminator(reversed_emb)

            return class_logits, attention_weights, slide_embedding, site_logits

else:
    class DomainAdversarialMIL:
        def __init__(self, *args, **kwargs): pass
