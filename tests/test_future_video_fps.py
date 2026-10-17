"""CPU regression checks for video timing metadata, without model weights."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from lingbotvla.data.data_collator import MakeMicroBatchCollator
from lingbotvla.data.multimodal.data_collator import VLADataCollatorWithPacking
from lingbotvla.data.vla_data.base_dataset import VLADataset
from lingbotvla.data.vla_data.utils import FeatureTransform
from lingbotvla.models.vla.vision_models.module_utils import get_video_target


class TokenizerStub:
    def __call__(self, prompts, **kwargs):
        return {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }


class FutureVideoFPSTest(unittest.TestCase):
    def test_dataset_to_teacher_boundary(self):
        robot = {
            "states": ["observation.state.arm.position"],
            "actions": [{"action.arm.position": {
                "origin_keys": "action", "subtract_state": False,
            }}],
            "images": ["observation.images.camera_top"],
            "norm_stats": None,
        }
        data = SimpleNamespace(joints=["{'arm.position': 2}"], cameras=["camera_top"])
        config = SimpleNamespace(max_state_dim=2, max_action_dim=2, tokenizer_max_length=4)
        processor = SimpleNamespace(tokenizer=TokenizerStub(), image_processor=None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.yaml"
            path.write_text(yaml.safe_dump(robot))
            transform = FeatureTransform(
                str(path), data, config, processor, do_nomalize=False,
                chunk_size=50, use_depth_align=True, use_future_image=True,
            )
            dataset = VLADataset.__new__(VLADataset)
            dataset.dataset = [{
                "observation.state.arm.position": torch.zeros(2),
                "action": torch.zeros(50, 2),
                "action_is_pad": torch.zeros(50, dtype=torch.bool),
                "observation.images.camera_top": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
                "task": "original",
            }]
            dataset.feature_transform = transform
            dataset.dataset_meta = SimpleNamespace(fps=30)
            dataset.chunk_size = 50
            dataset.use_future_image = True
            dataset.task_prompt = "override"
            dataset.transform = None
            sample = dataset.getitem(0)
            torch.testing.assert_close(sample['future_video_effective_fps'], torch.tensor(30 / 49))
            # Preserve explicitly supplied per-sample timing as well.
            dataset.dataset[0]['future_video_effective_fps'] = torch.tensor(2.0)
            second = dataset.getitem(0)
            batches = MakeMicroBatchCollator(1, VLADataCollatorWithPacking())([sample, second])
            batch = batches[0]
            fps = batch.pop('future_video_effective_fps')
            torch.testing.assert_close(fps, torch.tensor([30 / 49, 2.0]))

            class TeacherSpy:
                def get_future_feature(self, video, **kwargs):
                    torch.testing.assert_close(kwargs['fps'], fps)
                    self.called = True
                    return torch.zeros(2, 1, 1)

            teacher = TeacherSpy()
            get_video_target(
                teacher, batch['pil_images'], batch['future_pil_images'],
                {'input_size': 8, 'use_patch_loss': True}, effective_fps=fps,
            )
            self.assertTrue(teacher.called)
            # Ordinary samples without future images must not gain timing metadata.
            dataset.use_future_image = False
            del dataset.dataset[0]['future_video_effective_fps']
            self.assertNotIn('future_video_effective_fps', dataset.getitem(0))


if __name__ == '__main__':
    unittest.main()
