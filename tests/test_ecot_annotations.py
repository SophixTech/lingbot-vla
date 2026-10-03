from pathlib import Path
from unittest.mock import patch

import torch

from lingbotvla.data.vla_data.ecot_annotations import ECOTAnnotations


ANNOTATIONS = Path("/home/bjtc/Sophix/datasets/ecot_annotations/ecot1003")
DATASET = Path("/home/bjtc/Sophix/datasets/marked_9_1_0922")


def test_sample_alignment():
    annotations = ECOTAnnotations(ANNOTATIONS, split="train", source_dataset=DATASET)
    assert annotations.episode_indices == [0, 1]
    assert annotations.target(0, 0) == (0, [0.0] * 4, False, 1)
    assert annotations.target(0, 216)[0] == 1
    assert annotations.target(1, 278)[0] == 0
    assert annotations.target(1, 279)[0] == 2
    assert annotations.target(1, 451)[0] == 3
    assert annotations.target(1, 279)[2:] == (True, 0)


def test_bbox_loss_mask_with_no_valid_boxes():
    from lingbotvla.models.vla.pi0.modeling_lingbot_vla import FlowMatching

    model = FlowMatching.__new__(FlowMatching)
    torch.nn.Module.__init__(model)
    model.config = type("Config", (), {
        "n_action_steps": 2,
        "max_action_dim": 3,
        "align_params": {},
        "use_cache": False,
        "loss_type": "fm",
        "ecot_subtask_loss_weight": 0.1,
        "ecot_bbox_loss_weight": 0.05,
        "ecot_label_loss_weight": 0.1,
    })()
    model.ecot_aux_supervision = True
    model.use_depth_align = False
    model.ecot_subtask_head = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 4))
    model.ecot_bbox_head = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 4))
    model.ecot_label_head = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 2))
    model.action_out_proj = torch.nn.Linear(8, 3)

    class StubExpert(torch.nn.Module):
        def forward(self, inputs_embeds, **kwargs):
            return inputs_embeds, None

    model.qwenvl_with_expert = StubExpert()
    model.embed_prefix = lambda *args: (
        torch.randn(2, 4, 8, requires_grad=True),
        torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2, 4, dtype=torch.bool),
    )
    model.embed_suffix = lambda *args: (
        None,
        torch.randn(2, 3, 8, requires_grad=True),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 3, dtype=torch.bool),
    )
    model.sample_time = lambda batch_size, device: torch.full((batch_size,), 0.5, device=device)

    action_loss, _, _, aux_loss = model.forward(
        images=torch.empty(2), img_masks=torch.empty(2), lang_tokens=torch.empty(2),
        lang_masks=torch.empty(2), state=torch.zeros(2, 3), actions=torch.zeros(2, 2, 3),
        ecot_subtask_target=torch.tensor([0, 2]),
        ecot_bbox_target=torch.zeros(2, 4),
        ecot_bbox_valid=torch.tensor([False, False]),
        ecot_label_target=torch.tensor([1, 0]),
    )
    assert action_loss.shape == (2, 2, 3)
    assert torch.isfinite(aux_loss)
    (action_loss.mean() + aux_loss).backward()
    assert model.ecot_subtask_head[1].weight.grad is not None
    assert model.ecot_label_head[1].weight.grad is not None
    assert torch.count_nonzero(model.ecot_bbox_head[1].weight.grad) == 0

    model.zero_grad(set_to_none=True)
    _, _, _, aux_loss_with_bbox = model.forward(
        images=torch.empty(2), img_masks=torch.empty(2), lang_tokens=torch.empty(2),
        lang_masks=torch.empty(2), state=torch.zeros(2, 3), actions=torch.zeros(2, 2, 3),
        ecot_subtask_target=torch.tensor([0, 2]),
        ecot_bbox_target=torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0] * 4]),
        ecot_bbox_valid=torch.tensor([True, False]),
        ecot_label_target=torch.tensor([1, 0]),
    )
    aux_loss_with_bbox.backward()
    assert torch.count_nonzero(model.ecot_bbox_head[1].weight.grad) > 0


def test_old_checkpoint_initializes_only_new_heads():
    from lingbotvla.models.module_utils import load_model_weights

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"ecot_aux_supervision": True, "tie_word_embeddings": False})()
            self.model = torch.nn.Module()
            self.model.core = torch.nn.Linear(2, 2)
            for name, count in (("ecot_subtask_head", 4), ("ecot_bbox_head", 4), ("ecot_label_head", 2)):
                setattr(self.model, name, torch.nn.Sequential(torch.nn.LayerNorm(2), torch.nn.Linear(2, count)))

    model = Policy()
    core_weight = torch.ones_like(model.model.core.weight)
    core_bias = torch.zeros_like(model.model.core.bias)
    with patch("lingbotvla.models.module_utils._load_state_dict", return_value=[iter([
        ("model.core.weight", core_weight), ("model.core.bias", core_bias),
    ])]), patch("lingbotvla.models.module_utils.empty_cache"):
        load_model_weights(model, "unused", init_device="cpu", post_training=True)
    assert torch.equal(model.model.core.weight, core_weight)
    for head in (model.model.ecot_subtask_head, model.model.ecot_bbox_head, model.model.ecot_label_head):
        assert torch.isfinite(head[1].weight).all()
        assert torch.equal(head[0].weight, torch.ones_like(head[0].weight))
