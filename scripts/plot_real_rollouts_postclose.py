"""Plot the five specified real G1 rollouts after first right-gripper closure.

Only runtime_rollouts/1..5 are read.  Solid lines are recorded policy commands
(`aligned/steps.action`), dashed lines are the subsequent measured joint
states (`aligned/steps.measured_next_state`), in degrees.
"""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/bjtc/Sophix/g1/runtime_rollouts')
OUT = Path('/home/bjtc/Sophix/lingbot-vla/output/marked_full0919/real_rollouts_postclose_1_5')
N = 200
LABEL_DOWN_COLOR = '#d95f02'
LABEL_UP_COLOR = '#1f77b4'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = []
    # The original prompts ask for the object to be flipped: label down.
    sources = [(str(ep), ROOT / str(ep), 'label_down') for ep in range(1, 6)]
    # The changed prompts keep the label facing up: label up.
    sources += [(f'prompt_changed_{ep}', ROOT / f'prompt_changed_{ep}', 'label_up') for ep in range(1, 6)]
    for name, d, group in sources:
        meta = json.loads((d / 'episode.json').read_text())
        table = pq.read_table(d / 'aligned/steps.parquet').to_pydict()
        action = np.asarray(table['action'], dtype=np.float64)
        measured = np.asarray(table['measured_next_state'], dtype=np.float64)
        assert action.shape[1] == 16 and measured.shape == action.shape
        assert np.isfinite(action).all() and np.isfinite(measured).all()
        # Right-gripper command is dimension 15, normalized [0,1].
        gripper = action[:, 15]
        crossings = np.flatnonzero((gripper[1:] > 0.5) & (gripper[:-1] <= 0.5)) + 1
        if len(crossings) == 0:
            raise RuntimeError(f'rollout {name}: no right-gripper closure crossing')
        close = int(crossings[0])
        if close + N > len(action):
            raise RuntimeError(f'rollout {name}: only {len(action)-close} frames after closure')
        # The command at closure is frame 1 in the plotted post-close window.
        cmd = np.rad2deg(action[close:close+N, 7:14])
        obs = np.rad2deg(measured[close:close+N, 7:14])
        episodes.append(dict(episode=name, group=group, checkpoint=meta.get('checkpoint_id'),
                             norm_stats=meta.get('norm_stats_id'), control_hz=meta.get('control_hz'),
                             total_aligned_frames=len(action), closure_frame=close,
                             closure_crossings=[int(x) for x in crossings],
                             command_deg=cmd, measured_deg=obs,
                             right_arm_mae_deg=float(np.abs(cmd-obs).mean()),
                             right_arm_rmse_deg=float(np.sqrt(np.square(cmd-obs).mean())),
                             right_arm_max_abs_deg=float(np.abs(cmd-obs).max()),
                             per_joint_mae_deg=np.abs(cmd-obs).mean(axis=0).tolist(),
                             command_step_jitter_deg=float(np.abs(np.diff(cmd,axis=0)).mean()),
                             measured_step_jitter_deg=float(np.abs(np.diff(obs,axis=0)).mean())))

    colors = {'label_down': LABEL_DOWN_COLOR, 'label_up': LABEL_UP_COLOR}
    x = np.arange(1, N+1)
    fig, axes = plt.subplots(4, 2, figsize=(15, 17), sharex=True)
    axes = axes.ravel()
    for j in range(7):
        ax = axes[j]
        for rec in episodes:
            color = colors[rec['group']]
            label = str(rec['episode']).replace('prompt_changed_', 'prompt changed ')
            ax.plot(x, rec['command_deg'][:, j], color=color, lw=1.35,
                    label=label + ' command' if j == 0 else None)
            ax.plot(x, rec['measured_deg'][:, j], color=color, lw=1.15, ls='--', alpha=.9,
                    label=label + ' measured' if j == 0 else None)
        for boundary in (50.5, 100.5, 150.5):
            ax.axvline(boundary, color='0.65', lw=.7, ls=':')
        ax.set_title(f'Right joint {j+1}')
        ax.set_ylabel('Joint angle (deg)')
        ax.grid(alpha=.2)
    axes[7].axis('off')
    axes[6].set_xlabel('Frame after first right-gripper closure (1-200)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.005))
    fig.suptitle('Marked_full0919 real rollouts 1–5 + prompt-changed 1–5 | post-closure right-arm trajectories\nBlue: label up | Orange: label down | Solid: policy command | Dashed: measured next state', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / 'rollouts_1_5_postclose_right_arm.png', dpi=170)
    plt.close(fig)

    # Per-rollout panels make each experiment easier to inspect.
    for rec in episodes:
        color = colors[rec['group']]
        fig, axes = plt.subplots(4, 2, figsize=(13, 16), sharex=True)
        axes = axes.ravel()
        for j in range(7):
            ax = axes[j]
            ax.plot(x, rec['command_deg'][:, j], color=color, lw=1.5, label='policy command')
            ax.plot(x, rec['measured_deg'][:, j], color=color, lw=1.3, ls='--', alpha=.9, label='measured next state')
            for boundary in (50.5, 100.5, 150.5): ax.axvline(boundary, color='0.65', lw=.7, ls=':')
            ax.set_title(f'Right joint {j+1}'); ax.set_ylabel('deg'); ax.grid(alpha=.2)
        axes[7].axis('off'); axes[6].set_xlabel('Frame after closure (1-200)'); axes[6].legend(frameon=False)
        fig.suptitle(f'Rollout {rec["episode"]} | closure frame {rec["closure_frame"]}', fontsize=14)
        fig.tight_layout(); fig.savefig(OUT / f'rollout_{rec["episode"]}_postclose_right_arm.png', dpi=160); plt.close(fig)

    report = {
        'source_root': str(ROOT), 'included_rollouts': [1, 2, 3, 4, 5] + [f'prompt_changed_{i}' for i in range(1, 6)],
        'excluded_rollouts': 'All other directories',
        'checkpoint': episodes[0]['checkpoint'], 'norm_stats': episodes[0]['norm_stats'],
        'protocol': 'aligned/steps.parquet; first upward crossing of right-gripper action dimension 15 above 0.5; next 200 control frames; right arm dimensions 7:14; radians converted to degrees.',
        'line_semantics': 'solid=recorded policy command/action; dashed=measured_next_state aligned to that command; orange=label down; blue=label up',
        'episodes': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rec.items()} for rec in episodes],
    }
    (OUT / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('included_rollouts','checkpoint','norm_stats','line_semantics')}, indent=2))
    for rec in episodes:
        print(f"rollout {rec['episode']}: closure={rec['closure_frame']}, MAE={rec['right_arm_mae_deg']:.3f} deg, RMSE={rec['right_arm_rmse_deg']:.3f} deg, command jitter={rec['command_step_jitter_deg']:.3f} deg/frame, measured jitter={rec['measured_step_jitter_deg']:.3f} deg/frame")


if __name__ == '__main__':
    main()
