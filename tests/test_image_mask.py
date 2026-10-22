"""CPU checks for mask formats and actual training/deployment preprocessing hooks."""
import ast
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('image_mask', ROOT / 'lingbotvla/data/vla_data/image_mask.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RawImageMask = module.RawImageMask
CAM = 'observation.images.cam_high_rgb'


def method(path, cls, name, namespace):
    # Exercise actual entrypoint bodies without importing CUDA/model dependencies.
    tree = ast.parse((ROOT / path).read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    node = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class ImageMaskTest(unittest.TestCase):
    def mask(self, suffix='png'):
        return RawImageMask(SimpleNamespace(image_mask_enabled=True, image_mask_path=f'mask/data-v1/union_mask.{suffix}' if suffix == 'png' else 'mask/data-v1/union_ranges.json'))

    def test_formats_and_layouts(self):
        png, ranges = self.mask(), self.mask('json')
        np.testing.assert_array_equal(png.keep, ranges.keep)
        self.assertEqual(int(png.keep.sum()), 277540)
        source = np.full((480, 768, 3), 127, dtype=np.uint8)
        deployment = png.apply(source, CAM, channels_last=True)
        training = ranges.apply(torch.from_numpy(source.transpose(2, 0, 1)).float() / 255, CAM)
        np.testing.assert_allclose(training.numpy().transpose(1, 2, 0), deployment / 255, atol=1e-7)
        self.assertTrue(np.all(deployment[~png.keep] == 0))
        self.assertTrue(np.all(deployment[png.keep] == 127))
        self.assertTrue(np.all(source == 127))
        self.assertIs(png.apply(source, 'observation.images.cam_left_wrist_rgb', channels_last=True), source)

    def test_disabled_and_bad_resolution(self):
        source = np.ones((3, 224, 224))
        self.assertIs(RawImageMask(SimpleNamespace()).apply(source, CAM), source)
        with self.assertRaisesRegex(ValueError, 'original resolution'):
            self.mask().apply(source, CAM)
        with self.assertRaisesRegex(ValueError, 'requires image_mask_path'):
            RawImageMask(SimpleNamespace(image_mask_enabled=True))
        with self.assertRaises(FileNotFoundError):
            RawImageMask(SimpleNamespace(image_mask_enabled=True, image_mask_path='/missing/mask.png'))

    def test_invalid_json(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'mask.json'
            path.write_text(json.dumps(dict(width=2, height=1, rows=[[[0, 3]]])))
            with self.assertRaisesRegex(ValueError, 'Invalid mask interval'):
                RawImageMask(SimpleNamespace(image_mask_enabled=True, image_mask_path=str(path)))

    def test_training_masks_before_resize(self):
        getitem = method('lingbotvla/data/vla_data/base_dataset.py', 'LeRobotDataset', '__getitem__', {})
        mask = self.mask()
        seen = []
        source = torch.ones(3, 480, 768)
        dataset = SimpleNamespace(
            hf_dataset=[dict(episode_index=torch.tensor(0), task_index=torch.tensor(0), **{CAM: source})],
            delta_indices=None, load_image=True, raw_image_mask=mask,
            meta=SimpleNamespace(video_keys=[], camera_keys=[CAM], tasks=SimpleNamespace(iloc=[SimpleNamespace(name='task')])),
            image_transforms=lambda image: seen.append(image.clone()) or image)
        getitem(dataset, 0)
        self.assertEqual(tuple(seen[0].shape), (3, 480, 768))
        self.assertTrue(torch.all(seen[0][:, ~mask.keep] == 0))
        self.assertTrue(torch.all(source == 1))

    def test_deployment_masks_before_resize(self):
        resize = method('deploy/lingbot_vla_policy.py', 'LingbotVLAServer', 'resize_image', {'np': np, 'Image': Image})
        mask = self.mask()
        source = np.full((480, 768, 3), 127, dtype=np.uint8)
        server = SimpleNamespace(data_config=SimpleNamespace(img_size=224), vla=SimpleNamespace(
            feature_transform=SimpleNamespace(org_features={'images': [CAM]}, raw_image_mask=mask)))
        observation = {CAM: source}
        resize(server, observation)
        expected = np.asarray(Image.fromarray(mask.apply(source, CAM, channels_last=True)).resize((224, 224), Image.BILINEAR)).transpose(2, 0, 1) / 255
        np.testing.assert_array_equal(observation[CAM], expected)


if __name__ == '__main__':
    unittest.main()
