"""Offline, CPU-only checks for the sampler-level RTC primitive."""

import pytest
import torch

from lingbotvla.rtc import RTCConfig, RTCProcessor, RTCAttentionSchedule
from rtc_ablation import run as run_ablation


def _velocity(x):
    # A differentiable toy velocity with a non-trivial Jacobian.
    return 0.2 * x + 0.1


def test_first_chunk_is_unconstrained():
    x = torch.randn(1, 50, 16)
    processor = RTCProcessor()
    plain = _velocity(x)
    guided = processor.denoise_step(
        x_t=x, time=0.5, prev_chunk_left_over=None, inference_delay=20,
        original_velocity_fn=_velocity,
    )
    torch.testing.assert_close(guided, plain)


def test_prefix_transition_and_future_weights():
    processor = RTCProcessor()
    weights = processor.prefix_weights(20, 30, 50)
    assert torch.all(weights[:20] == 1)
    assert torch.all(weights[30:] == 0)
    assert torch.all(weights[20:30][:-1] > weights[20:30][1:])

    x = torch.zeros(1, 50, 16)
    previous = torch.ones(1, 50, 16)
    base = _velocity(x)
    guided = processor.denoise_step(
        x_t=x, time=0.5, prev_chunk_left_over=previous[:, :30], inference_delay=20,
        original_velocity_fn=_velocity,
    )
    delta = (guided - base).abs().mean(dim=(0, 2))
    assert delta[:20].mean() > delta[30:].mean()
    assert delta[30:].max() < 1e-6


def test_delay_index_and_model_space_shape_dtype_device():
    processor = RTCProcessor()
    x = torch.randn(1, 50, 16, dtype=torch.float32)
    previous = torch.randn(27, 16, dtype=torch.float32)
    out = processor.denoise_step(
        x_t=x, time=torch.tensor(0.7), prev_chunk_left_over=previous,
        inference_delay=22, original_velocity_fn=_velocity,
    )
    assert out.shape == x.shape
    assert out.dtype == x.dtype and out.device == x.device
    delta = (out - _velocity(x)).abs().mean(dim=(0, 2))
    assert delta[27:].max() < 1e-6


def test_offline_horizon_ablation_25_30_35():
    x = torch.zeros(1, 50, 16)
    previous = torch.ones(1, 50, 16)
    means = []
    for horizon in (25, 30, 35):
        processor = RTCProcessor()
        out = processor.denoise_step(
            x_t=x, time=0.5, prev_chunk_left_over=previous[:, :horizon],
            inference_delay=20,
            original_velocity_fn=_velocity,
        )
        delta = (out - _velocity(x)).abs().mean(dim=(0, 2))
        assert delta[horizon:].max() < 1e-6
        means.append(float(delta[:20].mean()))
    assert all(value > 0 for value in means)


def test_delay_indexing_values_have_valid_masks():
    processor = RTCProcessor()
    for delay in (0, 10, 20, 22, 25):
        weights = processor.prefix_weights(delay, 30, 50)
        assert weights.shape == (50,)
        assert torch.all(weights[:delay] == 1)
        assert torch.all(weights[30:] == 0)
        if delay < 30:
            assert torch.all((weights[delay:30] >= 0) & (weights[delay:30] <= 1))


def test_action_mask_blocks_padding_jacobian_from_valid_actions():
    processor = RTCProcessor()
    x = torch.zeros(1, 50, 4)
    previous = torch.ones(1, 50, 4)
    mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)

    def coupled_velocity(value):
        # Valid outputs depend on padding coordinates. Without the mask,
        # padding error would feed back into valid action coordinates.
        return value.flip(-1)

    base = coupled_velocity(x)
    guided = processor.denoise_step(
        x_t=x,
        time=0.5,
        prev_chunk_left_over=previous,
        inference_delay=0,
        action_mask=mask,
        original_velocity_fn=coupled_velocity,
    )
    delta = guided - base
    assert torch.any(delta[..., :2] != 0)
    torch.testing.assert_close(delta[..., 2:], torch.zeros_like(delta[..., 2:]))


def test_action_mask_preserves_guidance_on_active_dimensions():
    processor = RTCProcessor()
    x = torch.zeros(1, 50, 4)
    previous = torch.ones(1, 50, 4)
    mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)

    def identity_velocity(value):
        return 0.2 * value + 0.1

    base = identity_velocity(x)
    guided = processor.denoise_step(
        x_t=x,
        time=0.5,
        prev_chunk_left_over=previous,
        inference_delay=0,
        action_mask=mask,
        original_velocity_fn=identity_velocity,
    )
    delta = guided - base
    assert torch.any(delta[..., :2] != 0)
    torch.testing.assert_close(delta[..., 2:], torch.zeros_like(delta[..., 2:]))


def test_true_rtc_beats_independent_on_aligned_overlap_without_locking_future():
    results = run_ablation()
    independent = results["independent_sampling"]
    true = results["true_rtc"]
    assert true["aligned_overlap_mae"] < independent["aligned_overlap_mae"]
    assert true["boundary_action_delta"] < independent["boundary_action_delta"]
    # The post-horizon part remains observation-driven rather than copied from
    # the old trajectory.
    assert true["future_deviation_from_previous"] > 0


def test_default_exp_weights_match_paper_equation_and_official_code():
    processor = RTCProcessor()
    assert processor.config.prefix_attention_schedule == RTCAttentionSchedule.EXP
    start, end, total = 20, 25, 50
    weights = processor.prefix_weights(start, end, total)
    indices = torch.arange(total, dtype=torch.float32)
    c = torch.clamp((start - 1 - indices) / (end - start + 1) + 1, 0, 1)
    expected = c * torch.expm1(c) / (torch.e - 1)
    expected[indices >= end] = 0
    torch.testing.assert_close(weights, expected)


def test_delay_equal_to_overlap_is_valid_but_larger_delay_is_rejected():
    processor = RTCProcessor()
    x = torch.zeros(1, 50, 16)
    previous = torch.ones(1, 25, 16)
    out = processor.denoise_step(
        x_t=x, time=0.5, prev_chunk_left_over=previous,
        inference_delay=25, original_velocity_fn=_velocity,
    )
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match="exceeds previous-chunk overlap"):
        processor.denoise_step(
            x_t=x, time=0.5, prev_chunk_left_over=previous,
            inference_delay=26, original_velocity_fn=_velocity,
        )


def test_short_previous_chunk_never_guides_temporal_padding():
    processor = RTCProcessor()
    x = torch.zeros(1, 50, 4)
    previous = torch.ones(1, 25, 4)
    base = _velocity(x)
    guided = processor.denoise_step(
        x_t=x, time=0.5, prev_chunk_left_over=previous,
        inference_delay=22, original_velocity_fn=_velocity,
    )
    delta = (guided - base).abs()
    assert torch.any(delta[:, :25] > 0)
    torch.testing.assert_close(delta[:, 25:], torch.zeros_like(delta[:, 25:]))
