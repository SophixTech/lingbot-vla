from dataclasses import dataclass
from enum import Enum


class RTCAttentionSchedule(str, Enum):
    """Weight schedule for committed/transition/future action positions."""

    ZEROS = "zeros"
    ONES = "ones"
    LINEAR = "linear"
    EXP = "exp"


@dataclass(frozen=True)
class RTCConfig:
    """Inference-only RTC settings.  No checkpoint parameters are changed."""

    max_guidance_weight: float = 5.0
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP

    def __post_init__(self):
        if self.max_guidance_weight <= 0:
            raise ValueError("max_guidance_weight must be positive")
