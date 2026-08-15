"""Optimizer and learning-rate schedule configuration.

These are plain config dataclasses: the PyTorch trainer reads their fields and builds its
own `torch.optim.AdamW` and schedule from them. Nothing here constructs an optimizer.
"""

import dataclasses
from typing import Protocol


class LRScheduleConfig(Protocol):
    """Marker for learning-rate schedule configs."""


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 300
    vlm_warmup_steps: int = 800
    vlm_lr_multiplier: float = 0.1
    peak_lr: float = 5.5e-5
    decay_steps: int = 3_115
    decay_lr: float = 6.5e-6


class OptimizerConfig(Protocol):
    """Marker for optimizer configs."""


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0
