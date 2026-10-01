"""Deterministic synthetic ablation for the RTC acceptance metrics.

This is intentionally model-free: it exercises the same processor with a
known differentiable velocity field so regressions can run without a GPU or a
checkpoint.  Real checkpoint/observation replay remains a separate experiment.
"""
import json
import numpy as np
import torch

from lingbotvla.rtc import RTCConfig, RTCProcessor


def metrics(action, previous, independent, delay=20, horizon=30):
    action = np.asarray(action)
    d = np.diff(action, axis=0)
    dd = np.diff(action, n=2, axis=0)
    ddd = np.diff(action, n=3, axis=0)
    return {
        "aligned_overlap_mae": float(np.mean(np.abs(action[:delay] - previous[:delay]))),
        "boundary_action_delta": float(np.mean(np.abs(action[delay] - action[delay - 1]))),
        "velocity_discontinuity": float(np.mean(np.abs(d[delay] - d[delay - 1]))),
        "acceleration_discontinuity": float(np.mean(np.abs(dd[delay] - dd[delay - 1]))),
        "jerk_proxy": float(np.mean(np.abs(ddd[delay - 1]))),
        "future_trajectory_deviation": float(np.mean(np.abs(action[horizon:] - independent[horizon:]))),
        "future_deviation_from_previous": float(np.mean(np.abs(action[horizon:] - previous[horizon:]))),
    }


def run():
    steps, dim = 50, 16
    t = torch.linspace(0, 1, steps)[:, None]
    previous = torch.sin(t * np.pi).repeat(1, dim)
    independent = previous + 0.4 * torch.sin(t * 7 + torch.arange(dim)[None, :])
    # The toy field has a unit Jacobian and an independent observation-driven
    # target, making the effect of in-loop guidance easy to audit.
    x = independent.unsqueeze(0)
    def velocity(z):
        return z - independent.unsqueeze(0)
    processor = RTCProcessor()
    base = velocity(x)
    guided_velocity = processor.denoise_step(
        x_t=x, time=0.5, prev_chunk_left_over=previous[:30].unsqueeze(0),
        inference_delay=20, original_velocity_fn=velocity,
    )
    true_rtc = (x - 0.5 * guided_velocity)[0].detach().numpy()
    previous_np, independent_np = previous.numpy(), independent.numpy()
    blended = independent_np.copy()
    for i in range(5):
        w = (i + 1) / 5
        blended[20 + i] = (1 - w) * previous_np[20 + i] + w * blended[20 + i]
    return {
        "independent_sampling": metrics(independent_np, previous_np, independent_np),
        "async_skip": metrics(independent_np, previous_np, independent_np),
        "async_skip_plus_blend": metrics(blended, previous_np, independent_np),
        "true_rtc": metrics(true_rtc, previous_np, independent_np),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
