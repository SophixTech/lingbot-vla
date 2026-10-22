"""Fixed binary masks applied to raw camera frames before resizing."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


class RawImageMask:
    def __init__(self, config):
        self.enabled = getattr(config, 'image_mask_enabled', False)
        self.camera = getattr(config, 'image_mask_camera', 'observation.images.cam_high_rgb')
        self.keep = None
        if not self.enabled:
            return
        filename = getattr(config, 'image_mask_path', None)
        if not filename:
            raise ValueError('image_mask_enabled requires image_mask_path')
        path = Path(filename).expanduser()
        # Relative paths are always relative to the lingbot-vla checkout.
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        if path.suffix.lower() == '.json':
            data = json.loads(path.read_text())
            width, height = data['width'], data['height']
            if type(width) is not int or type(height) is not int or min(width, height) <= 0:
                raise ValueError('Mask width and height must be positive integers')
            if len(data['rows']) != height:
                raise ValueError('Mask rows must match height')
            keep = np.zeros((height, width), dtype=bool)
            for y, intervals in enumerate(data['rows']):
                for interval in intervals:
                    if len(interval) != 2:
                        raise ValueError('Mask intervals must contain [start, end)')
                    start, end = interval
                    if type(start) is not int or type(end) is not int or not 0 <= start < end <= width:
                        raise ValueError(f'Invalid mask interval at row {y}: {interval}')
                    keep[y, start:end] = True
            self.keep = keep
        else:
            with Image.open(path) as image:
                self.keep = np.array(image.convert('L')) > 0

    def apply(self, image, camera, *, channels_last=False):
        """Preserve dtype/range and source pixels; only masked pixels become zero."""
        if not self.enabled or camera != self.camera:
            return image
        shape = tuple(image.shape)
        spatial = shape[:2] if channels_last else shape[-2:]
        if len(shape) != 3 or spatial != self.keep.shape:
            raise ValueError(f'Mask for {camera} expects original resolution {self.keep.shape}, got {shape}')
        mask = self.keep[..., None] if channels_last else self.keep[None, ...]
        if isinstance(image, np.ndarray):
            return np.where(mask, image, np.zeros((), dtype=image.dtype))
        import torch
        return image.masked_fill(~torch.as_tensor(mask, device=image.device), 0)
