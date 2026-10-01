"""Flow-matching RTC guidance.

The processor is deliberately independent of a policy model.  It receives a
callable that predicts the original velocity and returns a velocity modified by
the RTC trajectory constraint *inside* every Euler denoising step.
"""

import math
from typing import Callable

import torch
from torch import Tensor

from .rtc_config import RTCConfig, RTCAttentionSchedule


class RTCProcessor:
    def __init__(self, config: RTCConfig | None = None):
        self.config = config or RTCConfig()

    def denoise_step(
        self,
        x_t: Tensor,
        prev_chunk_left_over: Tensor | None = None,
        inference_delay: int = 0,
        time: Tensor | float = 0.0,
        original_velocity_fn: Callable[[Tensor], Tensor] | None = None,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict and guide one flow-matching velocity.

        LingBot integrates from ``t=1`` to ``t=0`` with ``x += dt*v``.  Thus
        ``x_t - t*v`` is the model-space endpoint estimate.  The Jacobian
        transpose guidance is computed with autograd, matching the RTC paper
        and LeRobot implementation, rather than post-hoc action blending.
        """
        if original_velocity_fn is None:
            raise ValueError("original_velocity_fn is required")
        if prev_chunk_left_over is None:
            return original_velocity_fn(x_t)
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B,T,A], got {tuple(x_t.shape)}")
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, device=x_t.device, dtype=x_t.dtype)
            if mask.ndim == 1:
                mask = mask.view(1, 1, -1)
            elif mask.ndim == 2:
                mask = mask.unsqueeze(1)
            if mask.shape[-1] != x_t.shape[-1]:
                raise ValueError("action_mask dimension does not match x_t")
            if mask.shape[0] not in (1, x_t.shape[0]):
                raise ValueError("action_mask batch dimension does not match x_t")
            if mask.shape[0] == 1 and x_t.shape[0] != 1:
                mask = mask.expand(x_t.shape[0], -1, -1)
        else:
            mask = None
        if prev_chunk_left_over.ndim == 2:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)
        if prev_chunk_left_over.ndim != 3:
            raise ValueError("prev_chunk_left_over must be [B,T,A] or [T,A]")
        if prev_chunk_left_over.shape[0] not in (1, x_t.shape[0]):
            raise ValueError("previous chunk batch dimension does not match x_t")
        if prev_chunk_left_over.shape[0] == 1 and x_t.shape[0] != 1:
            prev_chunk_left_over = prev_chunk_left_over.expand(x_t.shape[0], -1, -1)
        if prev_chunk_left_over.shape[2] != x_t.shape[2]:
            raise ValueError("previous chunk action dimension does not match x_t")
        overlap_end = min(prev_chunk_left_over.shape[1], x_t.shape[1])
        if prev_chunk_left_over.shape[1] > x_t.shape[1]:
            prev_chunk_left_over = prev_chunk_left_over[:, : x_t.shape[1]]
        elif prev_chunk_left_over.shape[1] < x_t.shape[1]:
            pad = torch.zeros(
                x_t.shape[0], x_t.shape[1] - prev_chunk_left_over.shape[1], x_t.shape[2],
                device=x_t.device, dtype=x_t.dtype,
            )
            prev_chunk_left_over = torch.cat((prev_chunk_left_over.to(x_t), pad), dim=1)
        else:
            prev_chunk_left_over = prev_chunk_left_over.to(device=x_t.device, dtype=x_t.dtype)

        delay = int(inference_delay)
        if delay < 0:
            raise ValueError("inference_delay must be nonnegative")
        if delay > overlap_end:
            raise ValueError(
                f"inference_delay ({delay}) exceeds previous-chunk overlap ({overlap_end})"
            )
        # Algorithm 1 right-pads Aprev, but Eq. 5 ends the mask at H-s, which
        # is exactly the unpadded Aprev length. Padding must never be guided.
        weights = self.prefix_weights(delay, overlap_end, x_t.shape[1]).to(
            device=x_t.device, dtype=x_t.dtype
        )
        weights = weights.view(1, -1, 1)

        # ``select_action`` is normally under no_grad; RTC explicitly opens a
        # narrow autograd scope for the Jacobian guidance only.
        with torch.enable_grad():
            x = x_t.detach().requires_grad_(True)
            v = original_velocity_fn(x)
            t = torch.as_tensor(time, device=x.device, dtype=x.dtype)
            endpoint = x - t * v
            error = (prev_chunk_left_over - endpoint) * weights
            if mask is not None:
                error = error * mask
            correction = torch.autograd.grad(
                endpoint, x, grad_outputs=error.detach(), retain_graph=False, create_graph=False,
            )[0]
            if mask is not None:
                # Padded action coordinates are untrained.  Do not let their
                # Jacobian correction perturb valid robot action coordinates.
                correction = correction * mask

        # This time-dependent factor follows RTC's flow-matching guidance.  It
        # is clamped to keep late denoising steps numerically bounded.
        tau = 1 - t
        eps = torch.finfo(x.dtype).eps
        ratio = torch.nan_to_num((1 - tau) / tau.clamp_min(eps), posinf=self.config.max_guidance_weight)
        inv_r2 = ((1 - tau) ** 2 + tau**2) / (1 - tau).clamp_min(eps) ** 2
        guidance = torch.nan_to_num(ratio * inv_r2, posinf=self.config.max_guidance_weight)
        guidance = guidance.clamp(max=self.config.max_guidance_weight)
        return v.detach() - guidance * correction

    def prefix_weights(self, start: int, end: int, total: int) -> Tensor:
        end = max(0, min(int(end), total))
        start = min(max(0, min(int(start), total)), end)
        schedule = self.config.prefix_attention_schedule
        if schedule == RTCAttentionSchedule.ZEROS:
            out = torch.zeros(total)
            out[:start] = 1
            return out
        if schedule == RTCAttentionSchedule.ONES:
            out = torch.zeros(total)
            out[:end] = 1
            return out
        out = torch.zeros(total)
        out[:start] = 1
        n = end - start
        if n:
            linear = torch.linspace(1, 0, n + 2)[1:-1]
            if schedule == RTCAttentionSchedule.EXP:
                linear = linear * torch.expm1(linear) / (math.e - 1)
            out[start:end] = linear
        return out
