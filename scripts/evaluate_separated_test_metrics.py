"""Evaluate separated label-up/down checkpoints on held-out test views."""
from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from safetensors import safe_open
from transformers import AutoConfig
from lerobot.configs.policies import PreTrainedConfig

from deploy.lingbot_vla_policy import merge_qwen_config, set_seed_everywhere
from lingbotvla.data.vla_data.base_dataset import VLADataset
from lingbotvla.models import build_processor
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
from lingbotvla.utils.lora_utils import add_lora_to_model

ROOT = Path('/home/bjtc/Sophix/lingbot-vla')
DATA_ROOT = Path('/home/bjtc/Sophix/datasets/separated_data')
OUT_ROOT = ROOT / 'output/separated_model'
SPECS = (('label_up', 'label_up'), ('label_down', 'label_down'))


def latest_checkpoint(output_dir: Path) -> tuple[int, Path]:
    candidates = []
    for p in (output_dir / 'checkpoints').glob('global_step_*'):
        m = re.fullmatch(r'global_step_(\d+)', p.name)
        if not m:
            continue
        hf = p / 'hf_ckpt'
        if (hf / 'config.json').is_file() and (hf / 'model.safetensors.index.json').is_file():
            candidates.append((int(m.group(1)), hf))
    if not candidates:
        raise RuntimeError(f'No complete checkpoint under {output_dir / "checkpoints"}')
    return max(candidates)


def load_policy(checkpoint: Path):
    cfg = yaml.safe_load((checkpoint / 'lingbotvla_cli.yaml').read_text())
    set_seed_everywhere(42)
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.__dict__.update({k: v for k, v in {**cfg['model'], **cfg['train']}.items() if not hasattr(config, k)})
    config.attention_implementation = 'eager'
    config.tokenizer_path = cfg['model']['tokenizer_path']
    config = merge_qwen_config(config, AutoConfig.from_pretrained(config.tokenizer_path))
    config.use_cache = True
    processor = build_processor(config.tokenizer_path)
    policy = LingbotVlaPolicy(config, tokenizer_path=config.tokenizer_path)
    policy = add_lora_to_model(policy, lora_rank=cfg['train']['lora_rank'], lora_alpha=cfg['train']['lora_alpha'],
        lora_target_modules=cfg['train']['lora_target_modules'],
        lora_target_modules_support=('q_proj', 'k_proj', 'v_proj', 'o_proj'))
    weights = {}
    for shard in checkpoint.glob('*.safetensors'):
        with safe_open(shard, framework='pt', device='cpu') as f:
            weights.update({k: f.get_tensor(k) for k in f.keys()})
    policy.load_state_dict(weights, strict=True)
    del weights
    return cfg, config, processor, policy.to(device='cuda', dtype=torch.bfloat16).eval()


def add_metric(acc, name, error):
    error = np.asarray(error, dtype=np.float64)
    d = acc.setdefault(name, {'absolute_sum': 0.0, 'squared_sum': 0.0, 'count': 0})
    d['absolute_sum'] += float(np.abs(error).sum())
    d['squared_sum'] += float(np.square(error).sum())
    d['count'] += int(error.size)


def summarize(acc):
    result = {}
    for name, d in acc.items():
        mse = d['squared_sum'] / d['count']
        result[name] = {'mae': d['absolute_sum'] / d['count'], 'mse': mse,
                        'rmse': float(np.sqrt(mse)), 'scalar_count': d['count']}
    return result


def evaluate(name: str, dataset_name: str):
    output_dir = OUT_ROOT / name
    checkpoint_step, checkpoint = latest_checkpoint(output_dir)
    cfg, model_config, processor, policy = load_policy(checkpoint)
    data = SimpleNamespace(**cfg['data'])
    data.train_path = str(DATA_ROOT / dataset_name / 'test')
    data.exclude_regrasp = False
    ds = VLADataset(data.train_path, data.data_name, data, data.robot_config_root,
                    config=model_config, tokenizer=processor.tokenizer,
                    image_processor=processor.image_processor)
    jobs = []
    for episode in ds.dataset.meta.episodes:
        ep = int(episode['episode_index'])
        length = int(episode['length'])
        for fraction in (0.2, 0.5, 0.8):
            frame = min(length - 1, max(0, int((length - 1) * fraction)))
            jobs.append((ep, frame, int(episode['dataset_from_index']) + frame))

    acc = {}
    out_dir = output_dir / 'test_metrics'
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), (out_dir / 'samples.jsonl').open('w') as log:
        for job_id, (episode, frame, source_index) in enumerate(jobs):
            batch = ds.getdata(source_index)
            target = batch['actions'].clone()
            mask = batch['joint_mask']
            valid = ~batch['action_is_pad'].bool()
            args = [batch[key].unsqueeze(0).to(device='cuda', dtype=torch.bfloat16 if key in ('images', 'state') else batch[key].dtype)
                    for key in ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'state')]
            torch.manual_seed(10000 + job_id)
            torch.cuda.manual_seed_all(10000 + job_id)
            pred = policy.model.sample_actions(*args, num_steps=10).squeeze(0).float().cpu()
            nerr = (pred[valid][:, mask] - target[valid][:, mask]).double().numpy()
            gt = ds.feature_transform.unapply({**batch, 'actions': target.clone()})['action'].numpy()
            actual = ds.feature_transform.unapply({**batch, 'actions': pred.clone()})['action'].numpy()
            err = actual[valid.numpy()].astype(np.float64) - gt[valid.numpy()]
            if not np.isfinite(err).all() or not np.isfinite(nerr).all():
                raise RuntimeError(f'Non-finite prediction at episode={episode}, frame={frame}')
            add_metric(acc, 'normalized_16d', nerr)
            add_metric(acc, 'raw_action_16d_mixed_units', err)
            add_metric(acc, 'arm_14d', err[:, :14])
            add_metric(acc, 'right_arm_7d', err[:, 7:14])
            add_metric(acc, 'left_arm_7d', err[:, :7])
            add_metric(acc, 'gripper_2d', err[:, 14:])
            log.write(json.dumps({'episode': episode, 'frame': frame, 'source_index': source_index,
                                  'valid_frames': int(valid.sum())}) + '\n')
            if job_id % 30 == 0:
                print(f'{name}: evaluated {job_id + 1}/{len(jobs)}', flush=True)

    report = {'model': name, 'checkpoint': str(checkpoint), 'checkpoint_step': checkpoint_step,
              'dataset': data.train_path, 'sample_count': len(jobs),
              'protocol': 'Each test episode at 20%, 50%, and 80%; 50-action chunk; 10 denoising steps.',
              'metrics': summarize(acc), 'norm_stats_file': data.norm_stats_file,
              'norm_stats_sha256': hashlib.sha256(Path(data.norm_stats_file).read_bytes()).hexdigest()}
    (out_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    del policy, processor, ds
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    reports = {name: evaluate(name, dataset_name) for name, dataset_name in SPECS}
    (OUT_ROOT / 'test_metrics_summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == '__main__':
    main()
