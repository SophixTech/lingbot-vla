"""Unified-prompt data flow and a full-parameter training step with tiny real Qwen models."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml

from lingbotvla.data.data_transform import VLADataCollatorWithPacking
from lingbotvla.data.vla_data.base_dataset import PACKAGE_TASK_PROMPT, VLADataset
from lingbotvla.data.vla_data.ecot_annotations import ECOTAnnotations
from lingbotvla.models.vla.pi0 import modeling_lingbot_vla as modeling
from lingbotvla.optim import build_optimizer


@pytest.fixture
def annotations(tmp_path):
    source = tmp_path / "source"
    meta = source / "train/meta"
    meta.mkdir(parents=True)
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps({
        "episode_index": ep, "source_episode_id": f"source-{ep}", "frame_count": 2,
    }) for ep in range(2)))
    index = []
    for ep, label in enumerate((True, False)):
        record = {
            "grasp_label_up": label,
            "subtask": [
                {"start_frame": 0, "end_frame_exclusive": 1, "subtask": "pick_soft_package"},
                {"start_frame": 1, "end_frame_exclusive": 2,
                 "subtask": "place_label_up" if label else "flip_and_place_label_up"},
            ],
            "bbox": [[10, 20, 50, 60], None],
        }
        name = f"episode_{ep}.json"
        (tmp_path / name).write_text(json.dumps(record))
        index.append({"split": "train", "episode_index": ep, "source_episode_id": f"source-{ep}",
                      "frame_count": 2, "image_size": [100, 100], "annotation_path": name})
    # Other splits must not be validated against training metadata or loaded.
    index.append({"split": "test", "episode_index": 0, "annotation_path": "not_loaded.json"})
    (tmp_path / "annotations.jsonl").write_text("\n".join(map(json.dumps, index)))
    return ECOTAnnotations(tmp_path, "train", source)


def test_unified_prompt_and_sidecar_targets_survive_collation(annotations):
    dataset = VLADataset.__new__(VLADataset)
    dataset.ecot_annotations = annotations
    dataset.episode_labels = {}  # Sidecars also work without the older labels file.
    dataset.dataset = [
        {"episode_index": torch.tensor(ep), "frame_index": torch.tensor(frame), "task": "generic"}
        for ep in range(2) for frame in range(2)
    ]
    dataset.feature_transform = SimpleNamespace(apply=lambda item: item)
    items = [dataset.getdata(i) for i in range(4)]
    assert all(item["task"] == PACKAGE_TASK_PROMPT for item in items)
    batch = VLADataCollatorWithPacking()(items)
    assert batch["ecot_subtask_target"].tolist() == [0, 1, 0, 2]
    assert batch["ecot_label_target"].tolist() == [1, 1, 0, 0]
    assert batch["ecot_bbox_valid"].tolist() == [True, False, True, False]
    torch.testing.assert_close(batch["ecot_bbox_target"][0], torch.tensor([.1, .2, .5, .6]))
    assert annotations.target(999, 0) == (-100, [0., 0., 0., 0.], False, -100)


def test_unannotated_samples_keep_actions_and_mask_all_auxiliary_targets(annotations):
    dataset = VLADataset.__new__(VLADataset)
    dataset.ecot_annotations = annotations
    dataset.episode_labels = {
        2: {"extra_labels": {"self_annotation": {"grasp_label_up": False}}},
    }
    dataset.dataset = [
        {"episode_index": torch.tensor(ep), "frame_index": torch.tensor(0),
         "task": "source task", "actions": torch.full((2, 3), float(ep))}
        for ep in range(4)
    ]
    dataset.feature_transform = SimpleNamespace(apply=lambda item: item)
    batch = VLADataCollatorWithPacking()([dataset.getdata(i) for i in range(4)])
    assert batch["ecot_subtask_target"].tolist() == [0, 0, -100, -100]
    assert batch["ecot_label_target"].tolist() == [1, 0, -100, -100]
    assert batch["ecot_bbox_valid"].tolist() == [True, True, False, False]
    assert dataset.getdata(2)["task"] == dataset.getdata(1)["task"]
    assert all(dataset.getdata(i)["task"] == PACKAGE_TASK_PROMPT for i in range(4))
    torch.testing.assert_close(batch["actions"], torch.stack([
        torch.full((2, 3), float(ep)) for ep in range(4)]))


@pytest.mark.parametrize("config_name,output_dir", [
    ("package_ecot_full.yaml", "output/newarti_ecot_full"),
    ("package_ecot_fulltrain.yaml", "output/newarti_ecot_fulltrain"),
])
def test_fulltrain_config_includes_unannotated_episodes(config_name, output_dir):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/vla" / config_name).read_text())
    assert cfg["data"]["ecot_annotations_only"] is False
    assert cfg["data"]["exclude_regrasp"] is False
    assert Path(cfg["data"]["train_path"]).name == "train"
    assert cfg["train"]["ecot_aux_supervision"] is True
    assert cfg["train"]["output_dir"] == output_dir


def test_training_config_rejects_frozen_backbones():
    from tasks.vla.train_lingbotvla import validate_training_setup

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/vla/package_ecot_full.yaml").read_text())
    args = SimpleNamespace(train=SimpleNamespace(**cfg["train"]), data=SimpleNamespace(**cfg["data"]))
    validate_training_setup(args)
    for name, value in (("freeze_vision_encoder", True),
                        ("freeze_vit", True), ("train_expert_only", True), ("train_state_proj", False)):
        with patch.object(args.train, name, value), pytest.raises(ValueError):
            validate_training_setup(args)


def tiny_qwen_config(**kwargs):
    config = ORIGINAL_QWEN_CONFIG(**kwargs)
    for qwen in (config.qwenvl_config, config.qwen_expert_config):
        qwen.hidden_size = 32
        qwen.intermediate_size = 64
        qwen.num_hidden_layers = 2
        qwen.num_attention_heads = 4
        qwen.num_key_value_heads = 2
        qwen.head_dim = 8
        qwen.vocab_size = 128
    vision = config.qwenvl_config.vision_config
    vision.depth = 2
    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_heads = 4
    vision.out_hidden_size = 32
    vision.fullatt_block_indexes = [0, 1]
    return config


ORIGINAL_QWEN_CONFIG = modeling.QwenvlWithExpertConfig


@pytest.mark.skipif(
    not torch.cuda.is_available() or not modeling.is_flash_attn_available(),
    reason="Original LingBot attention requires CUDA and FlashAttention2",
)
@pytest.mark.parametrize("ecot_enabled", [True, False])
def test_full_parameter_step_and_conditioned_inference(ecot_enabled):
    torch.manual_seed(17)
    torch.set_num_threads(2)
    config = SimpleNamespace(
        freeze_vision_encoder=False, train_expert_only=False, train_state_proj=True,
        vocab_size=0, use_lm_head=False, attention_implementation="flex",
        tokenizer_path=None, use_cache=False, max_state_dim=3, max_action_dim=3,
        n_action_steps=2, action_dim=3, align_params={}, loss_type="L1_fm", num_steps=2,
        ecot_aux_supervision=ecot_enabled, ecot_action_conditioning=ecot_enabled, ecot_visibility=ecot_enabled,
    )
    # Only dimensions are reduced; the vision encoder, VLM, expert, heads,
    # action embedding, attention, objective, and optimizer are real modules.
    with patch.object(modeling, "QwenvlWithExpertConfig", tiny_qwen_config), patch.object(
        modeling.AutoConfig, "from_pretrained", return_value=tiny_qwen_config().qwenvl_config,
    ):
        flow = modeling.FlowMatching(config, eval=False)
    flow.to(device="cuda", dtype=torch.bfloat16)
    del flow.qwenvl_with_expert.qwenvl.lm_head
    del flow.qwenvl_with_expert.qwen_expert.lm_head
    policy = modeling.LingbotVlaPolicy.__new__(modeling.LingbotVlaPolicy)
    torch.nn.Module.__init__(policy)
    policy.config, policy.model = config, flow
    assert all(p.requires_grad for p in policy.parameters())
    if ecot_enabled:
        assert flow.ecot_condition_proj[0].in_features == 11
    optimizer = build_optimizer(policy, lr=1e-3, post_training=True)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in policy.parameters()}
    batch = dict(
        images=torch.randn(2, 3, 16, 3 * 2 * 14 * 14), img_masks=torch.ones(2, 3, dtype=torch.bool),
        lang_tokens=torch.randint(0, 128, (2, 5)), lang_masks=torch.ones(2, 5, dtype=torch.bool),
        state=torch.randn(2, 3), actions=torch.randn(2, 2, 3),
        ecot_subtask_target=torch.tensor([1, 2]), ecot_label_target=torch.tensor([1, 0]),
        ecot_bbox_target=torch.tensor([[.1, .2, .5, .6], [0., 0., 0., 0.]]),
        ecot_bbox_valid=torch.tensor([True, False]),
    )
    batch = {key: value.to(device="cuda", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
             for key, value in batch.items()}
    if not ecot_enabled:
        batch = {key: value for key, value in batch.items() if not key.startswith("ecot_")}
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    total, action, depth, logs, _ = policy(**batch)
    assert torch.isfinite(total)
    torch.testing.assert_close(total.detach(), action.detach() + depth + logs["ecot_aux_loss"])
    total.backward()
    groups = [
        "qwenvl.visual", "qwenvl.model", "qwen_expert.model", "state_proj", "action_out_proj",
    ]
    if ecot_enabled:
        groups += ["ecot_subtask_head", "ecot_bbox_head", "ecot_label_head", "ecot_visibility_head", "ecot_condition_proj"]
    else:
        assert logs["ecot_aux_loss"] == 0
    for group in groups:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for name, p in policy.named_parameters() if group in name), group
    optimizer.step()
    for group in groups:
        assert any(not torch.equal(before[name], p) for name, p in policy.named_parameters() if group in name), group
    policy.eval()
    # The deployment loader enables the prefix KV cache before sampling.
    config.use_cache = True
    with torch.no_grad():
        actions = flow.sample_actions(batch["images"], batch["img_masks"], batch["lang_tokens"],
                                      batch["lang_masks"], batch["state"], num_steps=2)
    assert actions.shape == (2, 2, 3) and torch.isfinite(actions).all()
