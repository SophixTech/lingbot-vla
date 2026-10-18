#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deploy.websocket_client_policy import WebsocketClientPolicy

ROOT = Path("/home/bjtc/Sophix")
ROLLOUT_ROOT = ROOT / "g1/runtime_rollouts"
OUT = ROOT / "worktrees/branch-ecot/output/ecot1003_conditioned_10000_runtime_replay_postclose_200"
N = 20
H = 200
COLORS = {"label_up": "#1f77b4", "label_down": "#d95f02"}
PROMPT = (
    "Take out the top soft package from the box; if the label is facing up, "
    "place it directly on the conveyor belt; if the label is facing down, "
    "flip it before placing it on the conveyor belt."
)
CHECKPOINT = Path("/var/tmp/bjtc_ecot1003_conditioned_20261009/checkpoints/global_step_10000/hf_ckpt")
NORM = Path("/home/bjtc/Sophix/datasets/marked_9_1_0922/train_stats/norm_stats.json")
CAM_KEYS = {
    "head": "observation.images.cam_high_rgb",
    "hand_left": "observation.images.cam_left_wrist_rgb",
    "hand_right": "observation.images.cam_right_wrist_rgb",
}


def rows(path: Path):
    return [json.loads(x) for x in (path / "steps.jsonl").read_text().splitlines() if x.strip()]


def obs_rows(path: Path):
    return [json.loads(x) for x in (path / "inference_observations.jsonl").read_text().splitlines() if x.strip()]


def label_for(i: int) -> str:
    return "label_down" if i <= 10 else "label_up"


def first_close(actions: np.ndarray) -> int:
    x = actions[:, 15] > 0.5
    c = np.flatnonzero(np.diff(x.astype(np.int8)) == 1) + 1
    if not len(c):
        raise RuntimeError("no right-gripper closure")
    return int(c[0])


def nearest_obs(observations, target_ns: int):
    return min(observations, key=lambda r: abs(int(r["capture_monotonic_ns"]) - target_ns))


def send_observation(client, root: Path, item: dict, prompt: str):
    payload = {"observation.state": np.asarray(item["state"], dtype=np.float32), "task": prompt}
    for name, key in CAM_KEYS.items():
        image = np.asarray(Image.open(root / item["camera_paths"][name]).convert("RGB"), dtype=np.uint8)
        payload[key] = image
    result = client.infer(payload)
    if "action" not in result:
        raise RuntimeError(f"server response has no action: {result.keys()}")
    return np.asarray(result["action"], dtype=np.float64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    client = WebsocketClientPolicy(host="127.0.0.1", port=8007)
    client.reset("package_a2d")
    predictions, truths, measured, labels, details = [], [], [], [], []
    for i in range(1, N + 1):
        root = ROLLOUT_ROOT / f"0930-{i}"
        sr = rows(root); ir = obs_rows(root)
        published = np.asarray([r["published_action"] for r in sr], dtype=np.float64)
        meas = np.asarray([r["measured_arm"] for r in sr], dtype=np.float64)
        close = first_close(published)
        if close + H > len(sr):
            raise RuntimeError(f"0930-{i}: fewer than {H} samples after closure")
        label = label_for(i); pred_chunks = []
        for k in range(4):
            target = int(sr[close + 50 * k]["capture_monotonic_ns"])
            item = nearest_obs(ir, target)
            chunk = send_observation(client, root, item, PROMPT)
            if chunk.shape[0] < 50 or chunk.shape[1] != 16:
                raise RuntimeError(f"0930-{i}: bad action shape {chunk.shape}")
            pred_chunks.append(chunk[:50])
        predictions.append(np.concatenate(pred_chunks))
        truths.append(published[close:close + H])
        measured.append(meas[close:close + H])
        labels.append(label)
        details.append({"rollout": f"0930-{i}", "label": label, "closure_step": close, "inference_observations": len(ir)})
        print(f"DONE 0930-{i} closure={close}", flush=True)
    predictions = np.asarray(predictions); truths = np.asarray(truths); measured = np.asarray(measured)
    np.savez_compressed(OUT / "paired_trajectories.npz", prediction=predictions, published=truths, measured=measured, label=np.asarray(labels))
    (OUT / "manifest.json").write_text(json.dumps({
        "checkpoint": str(CHECKPOINT), "norm_path": str(NORM), "prompt": PROMPT,
        "server": "ws://127.0.0.1:8007",
        "protocol": "Offline replay on observations recorded in 20 old mixed-conditioned physical rollouts; 4x50 chunks after their first gripper closure. This is not an ECOT closed-loop robot run.",
        "details": details}, indent=2) + "\n")
    plot(predictions, truths, measured, labels)


def plot(pred, truth, measured, labels):
    x = np.arange(1, H + 1)
    # Match standard post-closure comparison figure: group means, relative to
    # each episode's recorded action at the first closure.
    anchor = truth[:, :1, 7:14]
    pred_rel = np.rad2deg(pred[:, :, 7:14] - anchor)
    truth_rel = np.rad2deg(truth[:, :, 7:14] - anchor)
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=False)
    axes = axes.ravel()
    for j, ax in enumerate(axes[:7]):
        for label in ("label_up", "label_down"):
            mask = np.asarray(labels) == label
            color = COLORS[label]
            name = "up" if label == "label_up" else "down"
            ax.plot(x, pred_rel[mask, :, j].mean(axis=0), color=color, lw=1.8, label=f"{name} model")
            ax.plot(x, truth_rel[mask, :, j].mean(axis=0), color=color, lw=1.8, ls="--", label=f"{name} recorded")
        for b in (50.5, 100.5, 150.5): ax.axvline(b, color=".65", lw=.7, ls=":")
        ax.set_title(f"Right joint {j + 1}"); ax.set_ylabel("Change from closure (deg)")
        ax.set_xlabel("Frame after closure (1-200)"); ax.grid(alpha=.2)
    axes[7].axis("off")
    axes[0].legend(frameon=False, fontsize=9, ncol=2, loc="best")
    fig.suptitle("ecot1003 conditioned, global_step_10000: 20 recorded runtime observations (up=10, down=10)\n4 predictions; chunk=50; offline replay, not ECOT closed-loop", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "ecot1003_conditioned_postclose_200_runtime20_replay.png", dpi=170); plt.close(fig)


if __name__ == "__main__": main()
