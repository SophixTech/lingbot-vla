"""Compose independent LeRobot roots without merging files or episode IDs."""

import copy
import json
from pathlib import Path

from torch.utils.data import ConcatDataset

from .base_dataset import VLADataset


def resolve_train_paths(data_config):
    """An explicit list takes precedence over the legacy comma-separated string."""
    paths = getattr(data_config, 'train_paths', None)
    if paths is None:
        legacy = getattr(data_config, 'train_path', '')
        if not isinstance(legacy, str):
            raise ValueError('Use data.train_paths for a YAML list of dataset roots')
        paths = legacy.split(',')
    if not isinstance(paths, (list, tuple)) or not paths or any(not isinstance(p, str) or not p.strip() for p in paths):
        raise ValueError('Specify non-empty data.train_path or data.train_paths')
    paths = [str(Path(p.strip()).expanduser().resolve()) for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate training dataset roots (including symlink aliases)')
    return paths


def validate_dataset_interfaces(paths):
    """Reject incompatible numeric layouts before constructing or sampling datasets."""
    reference = None
    for path in paths:
        meta_path = Path(path) / 'meta/info.json'
        if not meta_path.is_file():
            raise ValueError(f'{path}: expected a LeRobot root containing meta/info.json; use train/ if split')
        info = json.loads(meta_path.read_text())
        if info.get('total_frames', 0) <= 0:
            raise ValueError(f'{path}: empty training dataset')
        signature = {
            'fps': info['fps'],
            'robot_type': info.get('robot_type'),
            'features': {
                key: {field: value.get(field) for field in ('dtype', 'shape', 'names')}
                for key, value in info['features'].items()
                if value['dtype'] != 'video'
            },
            # Spatial resolution may vary: the processor resizes each camera.
            'cameras': sorted(key for key, value in info['features'].items() if value['dtype'] == 'video'),
        }
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f'{path}: robot/FPS/numeric feature layout/camera keys differ from the first dataset')


def build_vla_dataset(data_config, **kwargs):
    """One child per root; the existing distributed sampler shuffles all frames.

    All children intentionally share one norm_stats_file and robot mapping.
    Each child retains its own metadata, episode IDs, videos, and action bounds.
    Sampling probability is proportional to retained frame count, not episode
    count or equal per-dataset weighting. Resume requires the same ordered roots.
    """
    paths = resolve_train_paths(data_config)
    validate_dataset_interfaces(paths)
    children = []
    for path in paths:
        child_config = copy.deepcopy(data_config)
        child_config.train_path = path
        child = VLADataset(
            repo_id=path, data_name=data_config.data_name,
            robot_config_root=data_config.robot_config_root,
            data_config=child_config,
            chunk_size=getattr(data_config, 'chunk_size', 50), **kwargs,
        )
        if not len(child):
            raise ValueError(f'{path}: no frames remain after episode filtering')
        children.append(child)
    dataset = children[0] if len(children) == 1 else ConcatDataset(children)
    dataset.training_sources = [
        {'path': path, 'frames': len(child), 'fraction': len(child) / sum(map(len, children))}
        for path, child in zip(paths, children)
    ]
    return dataset
