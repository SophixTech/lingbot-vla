"""Run the production policy forward with a stub backbone, without loading weights.

Extract only the method to avoid optional vision/FlashAttention imports. The
loss reduction under test is the actual method body, not a copied formula.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "lingbotvla/models/vla/pi0/modeling_lingbot_vla.py"
tree = ast.parse(SOURCE.read_text())
policy = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LingbotVlaPolicy")
forward = next(n for n in policy.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), forward], type_ignores=[])
namespace = {"torch": torch}
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)


def run_loss(losses, joint_mask=None, padding=None, action_dim=2, depth=0, action_loss_mask=None):
    stub = SimpleNamespace(
        config=SimpleNamespace(action_dim=action_dim, loss_type="mse"),
        model=SimpleNamespace(forward=lambda *args: (losses, depth, "depth-predictions")),
    )
    return namespace["forward"](
        stub, None, None, None, None, None, None,
        joint_mask=joint_mask, action_is_pad=padding, action_loss_mask=action_loss_mask,
    )


@pytest.mark.parametrize("with_joint_mask", [False, True])
def test_30_frame_episode_starting_at_frame_15(with_joint_mask):
    # 16 real timesteps, 34 padded timesteps. A third dimension is also padding.
    losses = torch.full((1, 50, 3), 1000.0)
    losses[:, :16, :2] = 2.0
    losses.requires_grad_()
    padding = torch.arange(50)[None, :] >= 16
    joints = torch.tensor([[True, True, False]]) if with_joint_mask else None
    total, action_loss, _, logs, _ = run_loss(losses, joints, padding)
    assert action_loss.item() == 2.0
    assert logs["batch_mean_losses"].tolist() == [2.0]
    assert logs["losses"].numel() == 32
    total.backward()
    expected = torch.zeros_like(losses)
    expected[:, :16, :2] = 1 / 32
    torch.testing.assert_close(losses.grad, expected)


def test_global_denominator_counts_valid_elements_and_preserves_depth():
    losses = torch.tensor([[[2., 100.], [50., 100.]], [[4., 4.], [4., 4.]]], requires_grad=True)
    joints = torch.tensor([[True, False], [True, True]])
    padding = torch.tensor([[False, True], [False, False]])
    total, action_loss, depth, logs, predictions = run_loss(losses, joints, padding, depth=torch.tensor(3.))
    assert action_loss.item() == pytest.approx(18 / 5)
    assert total.item() == pytest.approx(18 / 5 + 3)
    assert logs["batch_mean_losses"].tolist() == [2., 4.]
    assert depth.item() == 3 and predictions == "depth-predictions"


def test_no_time_padding_retains_joint_only_mean():
    losses = torch.tensor([[[1., 999., 3.], [5., 999., 7.]]])
    joints = torch.tensor([[True, False, True]])
    assert run_loss(losses, joints)[1].item() == 4.


@pytest.mark.parametrize("empty_by", ["time", "joint"])
def test_all_padding_has_finite_zero_loss_and_gradients(empty_by):
    losses = torch.ones((2, 50, 3), requires_grad=True)
    joints = torch.full((2, 3), empty_by != "joint")
    padding = torch.full((2, 50), empty_by == "time")
    total, action_loss, _, logs, _ = run_loss(losses, joints, padding)
    assert action_loss.item() == 0
    assert logs["batch_mean_losses"].tolist() == [0., 0.]
    total.backward()
    assert torch.count_nonzero(losses.grad) == 0


def test_invalid_mask_shapes_are_rejected():
    losses = torch.ones((2, 50, 3))
    with pytest.raises(ValueError, match="action_is_pad"):
        run_loss(losses, padding=torch.zeros(50, dtype=torch.bool))
    with pytest.raises(ValueError, match="joint_mask"):
        run_loss(losses, joint_mask=torch.ones((2, 50, 3), dtype=torch.bool))
    with pytest.raises(ValueError, match="action_loss_mask"):
        run_loss(losses, action_loss_mask=torch.ones(3, dtype=torch.bool))


def test_right_arm_mask_uses_padded_feature_offsets_and_zeroes_left_gradients():
    source = SOURCE.parents[3] / 'data/vla_data/utils.py'
    tree = ast.parse(source.read_text())
    transform = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FeatureTransform')
    method = next(n for n in transform.body if isinstance(n, ast.FunctionDef) and n.name == 'action_loss_mask')
    scope = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), scope)
    stub = SimpleNamespace(feature_config=SimpleNamespace(
        joints=['arm.position', 'effector.position'],
        joints_max_dim={'arm.position': 20, 'effector.position': 4},
    ))
    item = {'action.arm.position': torch.zeros(3, 14), 'action.effector.position': torch.zeros(3, 2)}
    joints = torch.zeros(32, dtype=torch.bool)
    joints[:14] = True
    joints[20:22] = True
    mask = scope['action_loss_mask'](stub, item, joints)
    assert mask.nonzero().flatten().tolist() == list(range(7, 14)) + [21]
    assert joints.sum() == 16  # structural mask remains intact

    predictions = torch.full((1, 3, 32), 2., requires_grad=True)
    targets = torch.zeros_like(predictions)
    targets[:, :, :7] = 10000  # Arbitrarily large left-arm errors must not contribute.
    losses = (predictions - targets).square()
    padding = torch.tensor([[False, False, True]])
    total, loss, _, logs, _ = run_loss(losses, joints[None], padding, action_loss_mask=mask[None])
    assert loss.item() == 4
    assert logs['losses'].numel() == 16
    total.backward()
    expected = torch.zeros_like(predictions)
    expected[:, :2, mask] = 4 / 16
    torch.testing.assert_close(predictions.grad, expected)

    item['action.arm.position'] = torch.zeros(3, 7)
    with pytest.raises(ValueError, match='14 dimensions'):
        scope['action_loss_mask'](stub, item, joints)
