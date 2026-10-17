"""Check the isolated V2 training runtime without loading model checkpoints."""
import argparse
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import site
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, help='Decode two frames from a local dataset video')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for key, directory in {
        'MPLCONFIGDIR': 'matplotlib',
        'TORCHINDUCTOR_CACHE_DIR': 'inductor',
        'TRITON_CACHE_DIR': 'triton',
    }.items():
        os.environ.setdefault(key, str(root / '.venv' / 'runtime-cache' / directory))
    # Allow direct invocation while keeping imports tied to this checkout.
    sys.path.insert(0, str(root))
    assert Path(sys.prefix).resolve() == root / '.venv', sys.prefix
    assert not site.ENABLE_USER_SITE
    old_root = root.parent / 'lingbot-vla'
    assert not any(Path(p).resolve().is_relative_to(old_root) for p in sys.path if p)
    import lingbotvla
    assert Path(lingbotvla.__file__).resolve().is_relative_to(root), lingbotvla.__file__
    print('Interpreter:', sys.executable, flush=True)
    print('Project:', lingbotvla.__file__, flush=True)
    expected = {
        'torch': '2.8.0+cu128', 'torchvision': '0.23.0+cu128',
        'torchaudio': '2.8.0+cu128', 'transformers': '4.57.3',
        'torchcodec': '0.6.0', 'datasets': '3.6.0', 'numpy': '1.26.4',
        'pyarrow': '21.0.0', 'lerobot': '0.4.2', 'accelerate': '1.7.0',
        'numpydantic': '1.8.1', 'plyfile': '1.1.2', 'flash-attn': '2.8.3',
    }
    from packaging.specifiers import SpecifierSet
    for name, version in expected.items():
        actual = metadata.version(name)
        assert SpecifierSet(f'=={version}').contains(actual), (name, actual, version)
        print(f'{name}=={actual}', flush=True)

    import numpy as np
    import pyarrow as pa
    from scipy.optimize import linear_sum_assignment
    np.testing.assert_array_equal(pa.array(np.arange(3)).to_numpy(), np.arange(3))
    assert linear_sum_assignment([[1, 5], [6, 2]])[1].tolist() == [0, 1]
    for name in [
        'torchvision', 'torchaudio', 'torchcodec.decoders', 'flash_attn',
        'lerobot.datasets.lerobot_dataset',
        'mdm.model.v2', 'moge.model.v2', 'moge.utils.vis',
        'lingbotvla.models.vla.vision_models.dino_video.teacher',
        'lingbotvla.models.vla.vision_models.dino_video.lumos_dinov3.models.navit_eval',
    ]:
        module = importlib.import_module(name)
        if name.startswith(('mdm.', 'moge.')):
            assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
        print('Import OK:', name, flush=True)

    from lingbotvla.models.vla.vision_models import module_utils
    assert hasattr(module_utils, 'v2') and hasattr(module_utils, 'v2_morgbd')
    if args.video:
        from torchcodec.decoders import VideoDecoder
        decoder = VideoDecoder(str(args.video), device='cpu', num_ffmpeg_threads=1)
        frames = decoder.get_frames_at(indices=[0, 1]).data
        assert frames.shape[0] == 2 and frames.shape[1] == 3, frames.shape
        print('Dataset video decode OK:', tuple(frames.shape), frames.dtype, flush=True)

    import torch
    if torch.cuda.is_available():
        # V2's grouped-GEMM decorators query device capability during import.
        for name in ['tasks.vla.train_lingbotvla', 'deploy.lingbot_vla_v2_policy']:
            importlib.import_module(name)
            print('Import OK:', name, flush=True)
        from flash_attn import flash_attn_func
        from torch.nn.attention.flex_attention import flex_attention
        print('GPU:', torch.cuda.get_device_name(), flush=True)
        q = torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        flash_attn_func(q, q, q).float().sum().backward()
        assert torch.isfinite(q.grad).all()
        f = torch.randn(1, 2, 16, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        torch.compile(flex_attention)(f, f, f).float().sum().backward()
        assert torch.isfinite(f.grad).all()
        print('GPU FlashAttention/FlexAttention forward/backward OK', flush=True)
    else:
        print('SKIPPED: training/deployment entry imports and GPU kernels require a visible CUDA device.', flush=True)
    print('PASS: available environment checks; full-model training capacity is a separate check.', flush=True)


if __name__ == '__main__':
    main()
