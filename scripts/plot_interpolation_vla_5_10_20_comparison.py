#!/usr/bin/env python3
"""Compare the 5/10/20 denoising-step LingBot-VLA training logs."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE = Path('/home/bjtc/Sophix/lingbot-vla/output')
RUNS = {
    '5 steps': BASE / 'interpolation_vla_num_denoising_step_5/checkpoints/loss.jsonl',
    '10 steps': BASE / 'interpolation_vla/checkpoints/loss.jsonl',
    '20 steps': BASE / 'interpolation_vla_num_denoising_step_20/checkpoints/loss.jsonl',
}
COLORS = {'5 steps': '#389e0d', '10 steps': '#1677ff', '20 steps': '#d4380d'}

def load(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

data = {name: load(path) for name, path in RUNS.items()}
n = min(len(v) for v in data.values())
x = np.arange(1, n + 1)
window = 100

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, layout='constrained')

for name, rows in data.items():
    rows = rows[:n]
    loss = np.array([r['loss'] for r in rows])
    grad = np.array([r['grad_norm'] for r in rows])
    c = COLORS[name]
    axes[0].plot(x, np.minimum(loss, 1.0), color=c, alpha=.20, linewidth=.7)
    axes[0].plot(x[window-1:], np.convolve(loss, np.ones(window)/window, mode='valid'), color=c, linewidth=1.8, label=f'{name} (100-step mean)')
    axes[1].plot(x, loss, color=c, alpha=.35, linewidth=.7, label=name)
    axes[2].plot(x, grad, color=c, alpha=.65, linewidth=.8, label=name)

axes[0].set_title(f'LingBot-VLA training comparison (shared prefix: {n} steps)')
axes[0].set_ylabel('Loss (clipped raw + 100-step mean)')
axes[0].set_ylim(0, 0.45)
axes[0].legend(ncol=3, fontsize=9)
axes[1].set_ylabel('Raw loss (symlog scale)')
axes[1].set_yscale('symlog', linthresh=0.1)
axes[1].legend(ncol=3, fontsize=9)
axes[2].set_ylabel('Gradient norm')
axes[2].set_xlabel('Training step')
axes[2].legend(ncol=3, fontsize=9)

out = BASE / 'interpolation_vla_num_denoising_step_20/vla_5_10_20_training_comparison.png'
fig.savefig(out, dpi=160, bbox_inches='tight')
print(out)
print('shared_steps', n)
