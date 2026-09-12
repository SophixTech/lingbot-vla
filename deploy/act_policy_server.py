#!/usr/bin/env python3
"""Serve a locally trained package ACT checkpoint over the native WebSocket API.

The checkpoint written by ``scripts/train_package_act.py`` contains the ACT
configuration and exact training mean/std statistics.  This server restores
both, so callers provide the same raw 16-D state and RGB observations used by
the G1 client and receive an unnormalized ``[50, 16]`` action chunk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from .websocket_policy_server import WebsocketPolicyServer


CAMERA_KEYS = (
    "observation.images.cam_high_rgb",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)
STATE_KEY = "observation.state"
ACTION_KEY = "action"


class ACTPolicyServer:
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"ACT checkpoint does not exist: {checkpoint_path}")

        # This is a locally generated training artifact, not an untrusted upload.
        checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {"policy", "policy_config", "stats"}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"ACT checkpoint is missing required fields: {sorted(missing)}")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for ACT inference but is unavailable")

        config = checkpoint["policy_config"]
        config.device = str(self.device)
        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(checkpoint["policy"], strict=True)
        self.policy.to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(config, dataset_stats=checkpoint["stats"])

        action_shape = tuple(config.output_features[ACTION_KEY].shape)
        self.chunk_size = int(config.chunk_size)
        self.action_dim = int(action_shape[-1])
        if self.chunk_size != 50 or self.action_dim != 16:
            raise ValueError(
                "This G1 package client requires an ACT checkpoint with action shape [50, 16], "
                f"got [{self.chunk_size}, {self.action_dim}]"
            )
        print(
            f"Loaded ACT step={checkpoint.get('step')} from {checkpoint_path}; "
            f"action_shape=[{self.chunk_size}, {self.action_dim}], device={self.device}",
            flush=True,
        )

    @staticmethod
    def _state(value: Any) -> np.ndarray:
        # msgpack may expose its receive buffer as read-only. Own the small
        # state array before converting it to a Torch tensor.
        state = np.array(value, dtype=np.float32, copy=True)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError(f"{STATE_KEY} must be a finite float32 vector with shape [16], got {state.shape}")
        return state

    @staticmethod
    def _image(key: str, value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
            raise ValueError(f"{key} must be a nonempty HxWx3 RGB image, got {image.shape}")
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError(f"{key} must contain finite numeric pixels")
        # GDK sends uint8 RGB. Accept float [0, 1] only for loopback/offline clients.
        if np.issubdtype(image.dtype, np.floating):
            if image.min() < 0.0 or image.max() > 1.0:
                raise ValueError(f"{key} float pixels must be in [0, 1]")
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
        return np.ascontiguousarray(np.moveaxis(image, -1, 0)).astype(np.float32) / 255.0

    def infer(self, observation: dict[str, Any]) -> dict[str, np.ndarray | None]:
        if observation.get("reset"):
            self.policy.reset()
            return {ACTION_KEY: None}

        # ACT's LeRobot preprocessor batches and moves Tensor inputs to its
        # configured device. Keep the wire format NumPy, but convert at this
        # boundary so the exact training processor path is reused.
        raw = {STATE_KEY: torch.from_numpy(self._state(observation.get(STATE_KEY)))}
        for key in CAMERA_KEYS:
            if key not in observation:
                raise KeyError(f"ACT observation is missing {key}")
            raw[key] = torch.from_numpy(self._image(key, observation[key]))

        with torch.inference_mode():
            normalized_batch = self.preprocessor(raw)
            normalized_action = self.policy.predict_action_chunk(normalized_batch)
            result = self.postprocessor(normalized_action)
        action = np.asarray(result, dtype=np.float32)
        if action.shape != (1, self.chunk_size, self.action_dim):
            raise RuntimeError(f"ACT postprocessor returned unexpected action shape: {action.shape}")
        action = action[0]
        if not np.isfinite(action).all():
            raise RuntimeError("ACT policy returned non-finite actions")
        return {ACTION_KEY: action}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8007)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")

    policy = ACTPolicyServer(args.checkpoint, device=args.device)
    server = WebsocketPolicyServer(
        policy,
        port=args.port,
        metadata={"policy": "act", "action_shape": [policy.chunk_size, policy.action_dim]},
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
