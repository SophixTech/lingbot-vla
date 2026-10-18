from pathlib import Path
from unittest.mock import patch
import json
import pytest

import torch

from lingbotvla.data.vla_data.ecot_annotations import ECOTAnnotations


ANNOTATIONS = Path("/home/bjtc/Sophix/datasets/ecot_annotations/ecot1003")
DATASET = Path("/home/bjtc/Sophix/datasets/marked_9_1_0922")


def test_index_without_dataset_path_rejects_wrong_source(tmp_path):
    source = tmp_path / 'source'
    meta = source / 'train/meta'
    meta.mkdir(parents=True)
    (meta / 'episodes.jsonl').write_text(json.dumps({
        'episode_index': 0, 'source_episode_id': 'actual', 'frame_count': 1,
    }) + '\n')
    (tmp_path / 'annotations.jsonl').write_text(json.dumps({
        'episode_index': 0, 'split': 'train', 'source_episode_id': 'wrong',
        'frame_count': 1,
    }) + '\n')
    with pytest.raises(ValueError, match='source episode identity/frame mismatch'):
        ECOTAnnotations(tmp_path, 'train', source)


def test_sample_alignment():
    annotations = ECOTAnnotations(ANNOTATIONS, split="train", source_dataset=DATASET)
    assert annotations.episode_indices
    for ep in annotations.episode_indices:
        record = annotations.get(ep)
        for frame in (0, len(record['bbox']) - 1):
            subtask, bbox, valid, label = annotations.target(ep, frame)
            assert subtask in range(4)
            assert valid == (record['bbox'][frame] is not None)
            assert all(0 <= coordinate <= 1 for coordinate in bbox)
            assert label == int(record['grasp_label_up'])


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

    # Action-only loss must reach all heads through the continuous condition,
    # even without any ground-truth labels supplied to the forward call.
    model.zero_grad(set_to_none=True)
    model.config.ecot_action_conditioning = True
    model.ecot_condition_proj = torch.nn.Sequential(torch.nn.Linear(10, 8))
    prefix = torch.randn(2, 4, 8, requires_grad=True)
    model.embed_prefix = lambda *args: (
        prefix, torch.ones(2, 4, dtype=torch.bool), torch.zeros(2, 4, dtype=torch.bool))
    action_loss, _, _, _ = model.forward(
        images=torch.empty(2), img_masks=torch.empty(2), lang_tokens=torch.empty(2),
        lang_masks=torch.empty(2), state=torch.zeros(2, 3), actions=torch.zeros(2, 2, 3))
    action_loss.mean().backward()
    for head in (model.ecot_subtask_head, model.ecot_label_head, model.ecot_bbox_head):
        assert head[1].weight.grad.abs().sum() > 0
    assert prefix.grad.abs().sum() > 0
    condition = model.ecot_condition(prefix.detach(), torch.ones(2, 4, dtype=torch.bool))
    torch.manual_seed(1)
    plain = model.predict_velocity(torch.zeros(2, 3), torch.ones(2, 4, dtype=torch.bool),
        None, torch.zeros(2, 2, 3), torch.ones(2))
    torch.manual_seed(1)
    conditioned = model.predict_velocity(torch.zeros(2, 3), torch.ones(2, 4, dtype=torch.bool),
        None, torch.zeros(2, 2, 3), torch.ones(2), condition)
    assert not torch.allclose(plain, conditioned)
    # Deployment computes conditions without label inputs and uses them on
    # every denoising step, including the RTC callback.
    seen = []
    def velocity(state, mask, cache, x, time, condition=None):
        assert condition is not None and condition.shape == (2, 8)
        seen.append(condition)
        return torch.zeros_like(x)
    model.predict_velocity = velocity
    model.sample_actions(torch.empty(2), torch.empty(2), torch.empty(2), torch.empty(2),
        torch.zeros(2, 3), noise=torch.zeros(2, 2, 3), num_steps=2)
    assert len(seen) == 2
    model.config.ecot_visibility = True
    model.ecot_visibility_head = torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 1))
    model.ecot_condition_proj = torch.nn.Sequential(torch.nn.Linear(11, 8))
    model.zero_grad(set_to_none=True)
    action, _, _, aux = model.forward(
        images=torch.empty(2), img_masks=torch.empty(2), lang_tokens=torch.empty(2),
        lang_masks=torch.empty(2), state=torch.zeros(2, 3), actions=torch.zeros(2, 2, 3),
        ecot_subtask_target=torch.tensor([0, 2]), ecot_label_target=torch.tensor([1, 0]),
        ecot_bbox_valid=torch.tensor([False, False]), ecot_bbox_target=torch.zeros(2, 4))
    aux.backward(retain_graph=True)
    assert model.ecot_visibility_head[1].weight.grad.abs().sum() > 0
    assert model.ecot_bbox_head[1].weight.grad.abs().sum() == 0
    model.zero_grad(set_to_none=True)
    action.mean().backward()
    assert model.ecot_visibility_head[1].weight.grad.abs().sum() > 0
    boxes = torch.tensor([[.1,.2,.3,.4],[.2,.3,.4,.5]])
    result = model.nullable_bboxes(boxes, torch.tensor([.1,.9]))
    assert result[0] is None and result[1] == boxes[1].tolist()


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


def test_label_probabilities_generate_trainable_layerwise_kv():
    from lingbotvla.models.vla.pi0.modeling_lingbot_vla import (
        EcotLabelKVProjector,
        FlowMatching,
    )

    model = FlowMatching.__new__(FlowMatching)
    torch.nn.Module.__init__(model)
    model.ecot_label_head = torch.nn.Sequential(
        torch.nn.LayerNorm(8), torch.nn.Linear(8, 2)
    )
    model.ecot_label_kv_projector = EcotLabelKVProjector(
        label_dim=2, hidden_size=8, num_layers=3, num_kv_heads=2, head_dim=4
    )
    prefix = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)

    layer_kv = model.ecot_label_kv(prefix, mask)
    assert len(layer_kv) == 3
    assert all(key.shape == value.shape == (2, 1, 2, 4) for key, value in layer_kv)

    sum(key.square().mean() + value.square().mean() for key, value in layer_kv).backward()
    assert model.ecot_label_head[1].weight.grad.abs().sum() > 0
    assert model.ecot_label_kv_projector.label_encoder[0].weight.grad.abs().sum() > 0
    assert model.ecot_label_kv_projector.layer_kv_proj[0].weight.grad.abs().sum() > 0


def test_extra_label_kv_is_visible_to_action_queries_only():
    from types import SimpleNamespace
    from lingbotvla.models.vla.pi0.modeling_lingbot_vla import QwenvlWithExpertModel
    from lingbotvla.models.vla.pi0.utils import our_eager_attention_forward

    masks_seen = []

    class Layer(torch.nn.Module):
        def __init__(self, action_expert=False):
            super().__init__()
            self.action_expert = action_expert

        def forward(self, hidden, att_output=None, start=0, end=0,
                    compute_kqv=False, output_atten=False, **kwargs):
            if compute_kqv:
                qkv = hidden.unsqueeze(2)
                if self.action_expert:
                    return qkv, qkv, qkv, None
                return qkv, qkv, qkv
            if output_atten:
                return hidden + att_output[:, start:end].reshape_as(hidden)
            raise AssertionError("unexpected layer call")

    class Backbone(torch.nn.Module):
        def __init__(self, action_expert=False):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(action_expert=action_expert)])
            self.norm = torch.nn.Identity()

    class VLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.config = SimpleNamespace(num_hidden_layers=1)

    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone(action_expert=True)

    model = QwenvlWithExpertModel.__new__(QwenvlWithExpertModel)
    torch.nn.Module.__init__(model)
    model.qwenvl = VLM()
    model.qwen_expert = Expert()
    model.config = SimpleNamespace(final_norm_adanorm=False)

    def attention(query, key, value, mask):
        masks_seen.append(mask.clone())
        return our_eager_attention_forward(query, key, value, mask)

    model.attention_interface = attention
    prefix = torch.randn(1, 2, 4)
    suffix = torch.randn(1, 1, 4)
    extra_key = torch.randn(1, 1, 1, 4)
    extra_value = torch.randn(1, 1, 1, 4, requires_grad=True)
    outputs, _ = model(
        attention_mask=torch.ones(1, 3, 3, dtype=torch.bool),
        position_ids=torch.tensor([[0, 1, 3]]),
        inputs_embeds=[prefix, suffix],
        use_cache=False,
        fill_kv_cache=False,
        extra_kv=[(extra_key, extra_value)],
        extra_kv_query_start=2,
    )
    assert masks_seen[0].shape == (1, 3, 4)
    assert masks_seen[0][0, :, -1].tolist() == [False, False, True]
    outputs[0].sum().backward(retain_graph=True)
    assert torch.count_nonzero(extra_value.grad) == 0
    outputs[1].sum().backward()
    assert extra_value.grad.abs().sum() > 0


def test_inference_appends_label_kv_once_and_reuses_cache():
    from lingbotvla.models.vla.pi0.modeling_lingbot_vla import (
        EcotLabelKVProjector,
        FlowMatching,
    )

    model = FlowMatching.__new__(FlowMatching)
    torch.nn.Module.__init__(model)
    model.config = type("Config", (), {
        "n_action_steps": 2, "max_action_dim": 3, "num_steps": 2,
        "use_cache": True, "ecot_action_conditioning": False,
        "ecot_label_kv_conditioning": True,
    })()
    model.use_depth_align = False
    model.ecot_label_head = torch.nn.Sequential(
        torch.nn.LayerNorm(8), torch.nn.Linear(8, 2)
    )
    model.ecot_label_kv_projector = EcotLabelKVProjector(
        label_dim=2, hidden_size=8, num_layers=2, num_kv_heads=1, head_dim=4
    )
    model.embed_prefix = lambda *args: (
        torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.bool),
    )

    class PrefixModel:
        def __init__(self):
            self.calls = 0

        def forward(self, inputs_embeds, **kwargs):
            self.calls += 1
            prefix = inputs_embeds[0]
            cache = {
                i: {
                    "key_states": torch.zeros(1, 3, 1, 4),
                    "value_states": torch.zeros(1, 3, 1, 4),
                }
                for i in range(2)
            }
            return [prefix, None], cache

    prefix_model = PrefixModel()
    model.qwenvl_with_expert = prefix_model
    seen = []

    def predict_velocity(state, prefix_mask, cache, x, timestep, condition=None):
        seen.append((prefix_mask.shape[1], tuple(cache[0]["key_states"].shape)))
        return torch.zeros_like(x)

    model.predict_velocity = predict_velocity
    actions = model.sample_actions(
        torch.empty(1), torch.ones(1, dtype=torch.bool), torch.empty(1),
        torch.empty(1), torch.zeros(1, 3), noise=torch.zeros(1, 2, 3),
    )
    assert actions.shape == (1, 2, 3)
    assert prefix_model.calls == 1
    assert seen == [(4, (1, 4, 1, 4)), (4, (1, 4, 1, 4))]
