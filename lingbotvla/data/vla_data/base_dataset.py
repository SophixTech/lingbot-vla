# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import json
from typing import Callable, Dict, List, Literal, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize
from transformers import AutoTokenizer
import json
import datasets
from collections.abc import Callable
from pathlib import Path

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from .utils import FeatureTransform
from .ecot_annotations import ECOTAnnotations


# Adapt branch-ecot's unified instruction to specify orientation before grasping.
PACKAGE_TASK_PROMPT = (
    "Take out the top soft package from the box. Based on its initial label "
    "orientation in the box before grasping: if the label initially faces up, "
    "place the package directly on the conveyor belt; if it initially faces down, "
    "flip the package so the label faces up before placing it on the conveyor belt."
)


class LeRobotDataset(BaseLeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        load_image: bool = True,
        **kwargs,
    ):
        super().__init__(repo_id, **kwargs)
        self.load_image = load_image

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """
        Query dataset for indices across keys, skipping video keys.

        Tries column-first [key][indices] for speed, falls back to row-first.

        Args:
            query_indices: Dict mapping keys to index lists to retrieve

        Returns:
            Dict with stacked tensors of queried data (video keys excluded)
        """
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            # Map absolute indices to relative indices if needed
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset.select(relative_indices)[key])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def __getitem__(self, idx) -> dict:
        # Ensure dataset is loaded when we actually need to read from it
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        
        query_indices = None
        if self.delta_indices is not None:
            # Episode bounds use source-global indices even in filtered views.
            query_indices, padding = self._get_query_indices(int(item["index"].item()), ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
            
        if len(self.meta.video_keys) > 0 and self.load_image:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None and self.load_image:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name

        return item
        

class VLADataset(Dataset):
    def __init__(
        self,
        repo_id,
        data_name,
        data_config,
        robot_config_root,
        config=PI0Config,
        tokenizer=AutoTokenizer,
        image_processor=None,
        video_backend = 'torchcodec',
        chunk_size = 50,
        image_size = (224, 224),
        do_nomalize = True,
        use_depth_align = False,
    ):

        self.image_processor = image_processor
        self.config = config
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.data_name = data_name

        self.ecot_annotations = None
        annotations_path = getattr(data_config, "ecot_annotations_path", None)
        if annotations_path:
            source = Path(repo_id)
            if source.name not in {"train", "test"}:
                raise ValueError("ECOT datasets must use a train/ or test/ split directory")
            self.ecot_annotations = ECOTAnnotations(
                annotations_path, split=source.name, source_dataset=source.parent,
            )
        elif getattr(data_config, "ecot_annotations_only", False):
            raise ValueError("ecot_annotations_only requires ecot_annotations_path")

        # The annotation run stores business labels separately from the
        # LeRobot action data.  Keep only non-regrasp demonstrations for the
        # normal behavior-cloning run when requested. Both initial label
        # orientations share the same instruction.
        self.episode_labels = {}
        labels_path = Path(repo_id) / "meta" / "annotations" / "episode_labels.jsonl"
        if labels_path.exists():
            with labels_path.open(encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    self.episode_labels[int(record["episode_index"])] = record
        self.selected_episodes = [
            ep for ep, record in sorted(self.episode_labels.items())
            if int(record.get("extra_labels", {}).get("self_annotation", {}).get("had_regrasp", 0)) == 0
        ] if self.episode_labels and getattr(data_config, 'exclude_regrasp', True) else None

        if self.ecot_annotations is not None:
            # Source annotations and auxiliary label supervision must agree.
            for ep in self.ecot_annotations.episode_indices:
                source_record = self.episode_labels.get(ep)
                if source_record is not None:
                    source_label = source_record["extra_labels"]["self_annotation"]["grasp_label_up"]
                    if source_label != self.ecot_annotations.get(ep)["grasp_label_up"]:
                        raise ValueError(f"Episode {ep}: source and ECOT grasp_label_up disagree")
            if getattr(data_config, "ecot_annotations_only", False):
                annotated = set(self.ecot_annotations.episode_indices)
                self.selected_episodes = sorted(
                    annotated if self.selected_episodes is None else annotated.intersection(self.selected_episodes)
                )
                if not self.selected_episodes:
                    raise ValueError("No ECOT-annotated episodes remain in this split")

        if do_nomalize:
            data_config.max_state_dim = self.config.max_state_dim
            data_config.max_action_dim = self.config.max_action_dim
            data_config.resize_imgs_with_padding = self.config.resize_imgs_with_padding
            data_config.tokenizer_max_length = self.config.tokenizer_max_length
        
        load_image = True if do_nomalize else False
        robot_config = os.path.join(robot_config_root, f'{data_name}.yaml')
        self.feature_transform = FeatureTransform(robot_config, data_config, \
                    tokenizer, image_processor, do_nomalize, \
                    chunk_size=chunk_size,use_depth_align=use_depth_align,
                    norm_stats_path=data_config.norm_stats_file,
                    load_image=load_image)

        self.action_features = self.feature_transform.actions
        self.state_features = self.feature_transform.states
        self.image_features = self.feature_transform.images
        
        self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        # ``load_image`` was removed from newer LeRobot releases. Keep the
        # project API stable while mapping it to the replacement
        # ``download_videos`` switch when needed.
        dataset_kwargs = dict(
            repo_id=repo_id,
            image_transforms=Resize(image_size),
            delta_timestamps=self.get_delta_timestamps(),
        )
        if self.selected_episodes is not None:
            dataset_kwargs["episodes"] = self.selected_episodes
        try:
            self.dataset = LeRobotDataset(**dataset_kwargs, load_image=load_image)
        except TypeError as exc:
            if "load_image" not in str(exc):
                raise
            self.dataset = LeRobotDataset(**dataset_kwargs, download_videos=load_image)

        if self.selected_episodes is None:
            # Full exports can exceed three-digit file numbering: lexical file
            # order places file-1000 before file-101. Action queries use original
            # global indices, so full-table row positions must match those IDs.
            self.dataset.hf_dataset = self.dataset.hf_dataset.sort('index')
            original_indices = np.asarray(self.dataset.hf_dataset.with_format('numpy')['index'])
            if not np.array_equal(original_indices, np.arange(len(original_indices))):
                raise ValueError('Full dataset must have contiguous original indices starting at zero')
        
        self.task_mapping = dict(zip(self.dataset_meta.tasks['task_index'], self.dataset_meta.tasks.index))

    def __len__(self):
        return len(self.dataset)

    def get_features(self):
        features = set()
        for feature_category, _features in self.feature_transform.org_features.items():
            features.update(_features)
        features.update(self.feature_transform.feature_to_keep)
        features = [x for x in list(features) if x not in ['action_is_pad', 'task']]
        return features

    def get_delta_timestamps(self):
        delta_timestamps = {}
        for action_feature in self.feature_transform.org_features['actions']:
                delta_timestamps[action_feature] = [t / self.dataset_meta.fps for t in range(self.chunk_size)]
        return delta_timestamps

    def getdata(self, idx):
        item = self.dataset[idx]
        # All episodes share the deployment instruction, with or without labels.
        item["task"] = PACKAGE_TASK_PROMPT
        ep_idx = int(item["episode_index"].item())
        if self.ecot_annotations is not None:
            subtask, bbox, bbox_valid, label_up = self.ecot_annotations.target(
                ep_idx, int(item["frame_index"].item()),
            )
            item["ecot_subtask_target"] = torch.tensor(subtask, dtype=torch.long)
            item["ecot_bbox_target"] = torch.tensor(bbox, dtype=torch.float32)
            item["ecot_bbox_valid"] = torch.tensor(bbox_valid, dtype=torch.bool)
            item["ecot_label_target"] = torch.tensor(label_up, dtype=torch.long)
        item = self.feature_transform.apply(item)
        return item

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                print(f'Error occurred while getting data {cur}: {str(e)}')
                last_err = e
                attempts += 1
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )
