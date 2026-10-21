"""Numerical contracts for ECOT's memory-efficient joint attention."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention


ROOT = Path(__file__).resolve().parents[1] / 'lingbotvla/models/vla/pi0'


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fast = load_module('ecot_flex', 'flex_attention.py')
eager = load_module('ecot_eager', 'utils.py').our_eager_attention_forward


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('with_label', [False, True])
@pytest.mark.parametrize('length', [17, 129])
def test_outputs_and_all_qkv_gradients(device, with_label, length, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; run on the training H200 for fused-kernel validation')
    if device == 'cpu':
        # Reference Flex execution checks score_mod semantics without a GPU compiler.
        monkeypatch.setattr(fast, '_flex_attention', flex_attention)
    torch.manual_seed(12)
    batch, heads, kv_heads, width = 2, 4, 2, 16
    key_length = length + int(with_label)
    values = [
        torch.randn(batch, length, heads, width, device=device),
        torch.randn(batch, key_length, kv_heads, width, device=device),
        torch.randn(batch, key_length, kv_heads, width, device=device),
    ]
    mask = torch.ones(batch, length, key_length, dtype=torch.bool, device=device)
    # Bidirectional prefix, suffix reads prefix; prefix cannot read suffix/label.
    mask[:, :11, 11:] = False
    mask[0, -1, :] = False  # padding: eager returns uniform values, not zeros
    mask[1, :, 3] = False
    ref_inputs = [x.detach().clone().requires_grad_() for x in values]
    new_inputs = [x.detach().clone().requires_grad_() for x in values]
    ref = eager(*ref_inputs, mask)
    new = fast.flex_attention_forward(*new_inputs, mask)
    torch.testing.assert_close(new, ref, atol=2e-5, rtol=2e-4)
    upstream = torch.randn_like(ref)
    ref_grads = torch.autograd.grad((ref * upstream).sum(), ref_inputs)
    new_grads = torch.autograd.grad((new * upstream).sum(), new_inputs)
    for actual, expected in zip(new_grads, ref_grads):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    if with_label:
        # Label keys/values must remain trainable via suffix queries.
        assert new_grads[1][:, -1].abs().sum() > 0
        assert new_grads[2][:, -1].abs().sum() > 0
        changed = [x.detach().clone() for x in values]
        changed[1][:, -1] += 10
        changed[2][:, -1] -= 10
        altered = fast.flex_attention_forward(*changed, mask)
        torch.testing.assert_close(altered[:, :11], new[:, :11], atol=2e-5, rtol=2e-4)


def test_dynamic_score_mod_captures_only_tensor_buffers(monkeypatch):
    """Regression: CUDA autotuning cannot accept SymInt score-mod buffers."""
    monkeypatch.setattr(fast, '_flex_attention', flex_attention)
    captured = []

    def inspect_backend(graph, example_inputs):
        for node in graph.graph.nodes:
            if node.op == 'call_function' and node.target == torch.ops.higher_order.flex_attention:
                buffers = node.args[7]
                assert buffers
                for buffer in buffers:
                    assert isinstance(buffer, torch.fx.Node)
                    assert isinstance(buffer.meta['example_value'], torch.Tensor)
                captured.append(len(buffers))
        return graph.forward

    compiled = torch.compile(fast.flex_attention_forward, backend=inspect_backend,
                             dynamic=True, fullgraph=True)
    for query_len, key_len in [(17, 17), (17, 18), (19, 20)]:
        q = torch.randn(2, query_len, 4, 16, requires_grad=True)
        k = torch.randn(2, key_len, 2, 16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        mask = torch.ones(2, query_len, key_len, dtype=torch.bool)
        mask[:, :11, 11:] = False
        actual = compiled(q, k, v, mask)
        expected = eager(q, k, v, mask)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        actual_grads = torch.autograd.grad(actual.sum(), (q, k, v))
        expected_grads = torch.autograd.grad(expected.sum(), (q, k, v))
        for a, e in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, e, atol=3e-5, rtol=3e-4)
    assert captured
