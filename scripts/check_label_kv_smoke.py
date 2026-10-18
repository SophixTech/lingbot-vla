"""Validate a short-run checkpoint offline; never connect to a robot."""
import argparse
import json
import os
from pathlib import Path

import torch

from deploy.lingbot_vla_policy import LingbotVLAServer
from lingbotvla.data.vla_data.base_dataset import VLADataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    os.environ.setdefault("QWEN25_PATH", "/home/bjtc/Sophix/models/Qwen2.5-VL-3B-Instruct-tokenizer")
    torch.set_num_threads(2)
    server = LingbotVLAServer(path_to_pi_model=args.checkpoint, use_compile=False)
    policy = server.vla
    model = policy.model
    assert model.config.ecot_label_kv_conditioning
    assert not model.config.ecot_action_conditioning
    data = server.data_config
    dataset = VLADataset(
        repo_id=data.train_path, data_name=data.data_name, data_config=data,
        robot_config_root=data.robot_config_root, config=policy.config,
        tokenizer=server.processor.tokenizer,
        image_processor=server.processor.image_processor,
    )
    item = dataset[0]
    inputs = [item[name].unsqueeze(0).cuda().to(
        torch.bfloat16 if name in ("images", "state") else item[name].dtype
    ) for name in ("images", "img_masks", "lang_tokens", "lang_masks", "state")]
    calls = {"kv": 0, "prefix": 0}
    def count_kv(*unused):
        calls["kv"] += 1
    original_forward = model.qwenvl_with_expert.forward
    def count_prefix(*positional, **kwargs):
        if kwargs.get("inputs_embeds", [None])[0] is not None:
            calls["prefix"] += 1
        return original_forward(*positional, **kwargs)
    h1 = model.ecot_label_kv_projector.register_forward_hook(count_kv)
    model.qwenvl_with_expert.forward = count_prefix
    with torch.inference_mode():
        prediction = model.sample_actions(*inputs, num_steps=10)
    h1.remove()
    model.qwenvl_with_expert.forward = original_forward
    assert torch.isfinite(prediction).all()
    assert calls == {"kv": 1, "prefix": 1}, calls
    # Isolate the action-loss gradient; no auxiliary ground truth is supplied.
    policy.requires_grad_(False)
    model.ecot_label_head.requires_grad_(True)
    model.ecot_label_kv_projector.requires_grad_(True)
    actions = item["actions"].unsqueeze(0).cuda().to(torch.bfloat16)
    loss, _, _, _ = model(*inputs, actions=actions)
    loss.mean().backward()
    gradients = [float(layer.weight.grad.float().norm())
                 for layer in model.ecot_label_kv_projector.layer_kv_proj]
    assert len(gradients) == 36
    assert all(value > 0 and value < float("inf") for value in gradients)
    label_grad = float(model.ecot_label_head[1].weight.grad.float().norm())
    assert label_grad > 0 and label_grad < float("inf")
    result = {
        "checkpoint": args.checkpoint,
        "training_episode_count": len(dataset.selected_episodes),
        "training_frame_count": len(dataset),
        "strict_deployment_load": True,
        "prediction_shape": list(prediction.shape),
        "prediction_finite": True,
        "denoising_steps": 10,
        "calls_per_observation": calls,
        "action_only_loss": float(loss.mean().detach()),
        "label_head_grad_norm": label_grad,
        "layer_kv_grad_norms": gradients,
        "robot_connected": False,
    }
    output = Path(args.checkpoint).parents[2] / "smoke_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
