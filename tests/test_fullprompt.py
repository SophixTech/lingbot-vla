"""Regression checks for mixed-orientation training with one instruction."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from lingbotvla.data.vla_data.base_dataset import PACKAGE_TASK_PROMPT, VLADataset


class FullPromptTest(unittest.TestCase):
    def test_both_orientations_and_missing_annotations_share_instruction(self):
        dataset = VLADataset.__new__(VLADataset)
        dataset.ecot_annotations = None
        dataset.episode_labels = {
            0: {"extra_labels": {"self_annotation": {"grasp_label_up": True}}},
            1: {"extra_labels": {"self_annotation": {"grasp_label_up": False}}},
        }
        dataset.dataset = [
            {
                "episode_index": torch.tensor(index),
                "task": task,
                "action": torch.full((50, 16), float(index)),
                "observation.state": torch.full((16,), float(index)),
            }
            for index, task in enumerate(("keep label up", "flip package", "source task"))
        ]
        observed_prompts = []

        def capture_transform(item):
            observed_prompts.append(item["task"])
            return item

        dataset.feature_transform = SimpleNamespace(apply=capture_transform)
        for index in range(len(dataset)):
            with self.subTest(episode=index):
                item = dataset[index]
                self.assertEqual(item["task"], PACKAGE_TASK_PROMPT)
                self.assertTrue(torch.equal(item["action"], torch.full((50, 16), float(index))))
                self.assertTrue(torch.equal(item["observation.state"], torch.full((16,), float(index))))
        self.assertEqual(observed_prompts, [PACKAGE_TASK_PROMPT] * 3)

    def test_documented_deployment_instruction_matches_loader(self):
        root = Path(__file__).resolve().parents[1]
        contract = json.loads((root / "configs/package_a2d_contract.json").read_text())
        self.assertEqual(contract["task"]["instruction"], PACKAGE_TASK_PROMPT)


if __name__ == "__main__":
    unittest.main()
