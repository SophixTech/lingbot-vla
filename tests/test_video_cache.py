"""Resource lifetime and timestamp routing regressions for VLA video loading."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tempfile
import unittest
import torch

from lingbotvla.data.vla_data import video_cache as vc
from lingbotvla.data.vla_data.base_dataset import LeRobotDataset


class VideoCacheTest(unittest.TestCase):
    def setUp(self):
        vc.clear_video_cache()
        self.directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.directory.name)

    def tearDown(self):
        vc.clear_video_cache()
        self.directory.cleanup()

    def test_lru_eviction_closes_file_after_decoder_destruction(self):
        tmp_path = self.tmp_path
        events = []
        handles = []

        class Decoder:
            def __init__(self, handle, **kwargs):
                self.handle = handle
                handles.append(handle)
            def __del__(self):
                events.append((Path(self.handle.name).name, self.handle.closed))

        for name in ('a', 'b', 'c'):
            (tmp_path / name).write_bytes(b'fake')
        with patch('torchcodec.decoders.VideoDecoder', Decoder):
            cache = vc.BoundedVideoDecoderCache(2)
            cache.get_decoder(tmp_path / 'a')
            cache.get_decoder(tmp_path / 'b')
            cache.get_decoder(tmp_path / 'a')  # a is now most recently used
            cache.get_decoder(tmp_path / 'c')
            assert cache.size() == 2
            assert events == [('b', False)]
            assert handles[1].closed and not handles[0].closed
            cache.clear()
            assert all(h.closed for h in handles)
            assert all(not closed for _, closed in events)


    def test_failed_decoder_creation_closes_file(self):
        tmp_path = self.tmp_path
        path = tmp_path / 'bad'; path.write_bytes(b'fake')
        handles = []
        def fail(handle, **kwargs):
            handles.append(handle)
            raise ValueError('broken video')
        cache = vc.BoundedVideoDecoderCache(2)
        with patch('torchcodec.decoders.VideoDecoder', side_effect=fail):
            with self.assertRaisesRegex(ValueError, 'broken video'):
                cache.get_decoder(path)
        assert cache.size() == 0 and handles[0].closed


    def test_two_roots_share_cache_and_respect_timestamp_offsets(self):
        tmp_path = self.tmp_path
        seen = []
        def decode(path, times, tolerance, decoder_cache):
            seen.append((path, times, tolerance, decoder_cache))
            return torch.zeros(len(times), 3, 2, 2, dtype=torch.uint8)
        with patch('lingbotvla.data.vla_data.video_utils.decode_video_frames_torchcodec', side_effect=decode):
            for name, offset in [('old', 5.0), ('new', 10.0)]:
                ds = object.__new__(LeRobotDataset)
                ds.root = tmp_path / name
                ds.video_backend = 'torchcodec'; ds.video_decoder_cache_size = 2; ds.tolerance_s = .04
                ds.meta = SimpleNamespace(episodes=[{'videos/cam/from_timestamp': offset}],
                                          get_video_file_path=lambda ep, key: Path('video.mp4'))
                assert ds._query_videos({'cam': [.5, 1.5]}, 0)['cam'].shape == (2, 3, 2, 2)
        assert seen[0][1] == [5.5, 6.5] and seen[1][1] == [10.5, 11.5]
        assert seen[0][3] is seen[1][3]
        assert seen[0][0] != seen[1][0]


    def test_invalid_capacity_and_process_reset(self):
        with self.assertRaises(ValueError):
            vc.BoundedVideoDecoderCache(0)
        cache = vc._get_cache(2)
        vc._reset_after_fork()
        assert vc._get_cache(2) is not cache

    def test_other_backend_uses_original_decoder(self):
        ds = object.__new__(LeRobotDataset)
        ds.root = self.tmp_path
        ds.video_backend = 'pyav'
        ds.tolerance_s = .04
        ds.meta = SimpleNamespace(
            episodes=[{'videos/cam/from_timestamp': 5.0}],
            get_video_file_path=lambda ep, key: Path('video.mp4'),
        )
        with patch('lingbotvla.data.vla_data.base_dataset.decode_video_frames') as decode:
            decode.return_value = torch.zeros(1, 3, 2, 2)
            ds._query_videos({'cam': [.5]}, 0)
            decode.assert_called_once_with(self.tmp_path / 'video.mp4', [5.5], .04, 'pyav')
