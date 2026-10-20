"""Bounded, process-local torchcodec cache shared by all dataset roots.

Keep v2 timestamp validation, frame clamping and uint8 output unchanged,
while reusing a bounded decoder cache. Create decoders
lazily inside workers; no decoder or lock is stored on the pickled Dataset.
"""
import atexit
from collections import OrderedDict
import os
import threading


class BoundedVideoDecoderCache:
    def __init__(self, capacity=32):
        if capacity < 1:
            raise ValueError('video_decoder_cache_size must be positive')
        self.capacity = capacity
        self._cache = OrderedDict()

    def _evict(self):
        _, (decoder, handle) = self._cache.popitem(last=False)
        # Destroy the decoder while its input file is still valid.
        del decoder
        handle.close()

    def get_decoder(self, video_path):
        from torchcodec.decoders import VideoDecoder

        key = os.path.abspath(os.fspath(video_path))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        # Evict BEFORE opening the next video so even misses respect the bound.
        while len(self._cache) >= self.capacity:
            self._evict()
        handle = open(key, 'rb')
        try:
            decoder = VideoDecoder(handle, seek_mode='approximate', num_ffmpeg_threads=1)
        except BaseException:
            handle.close()
            raise
        self._cache[key] = (decoder, handle)
        return decoder

    def clear(self):
        while self._cache:
            self._evict()

    def size(self):
        return len(self._cache)


_cache = None
_lock = threading.RLock()


def _reset_after_fork():
    global _cache, _lock
    if _cache is not None:
        _cache.clear()
    _cache = None
    _lock = threading.RLock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _get_cache(capacity):
    global _cache
    if capacity < 1:
        raise ValueError('video_decoder_cache_size must be positive')
    if _cache is None:
        _cache = BoundedVideoDecoderCache(capacity)
    else:
        _cache.capacity = capacity
        while _cache.size() > capacity:
            _cache._evict()
    return _cache


def decode_video_frames_bounded(video_path, timestamps, tolerance_s, capacity=32):
    from .video_utils import decode_video_frames_torchcodec

    # Keep a decoder alive for the entire frame query even if called by threads.
    with _lock:
        return decode_video_frames_torchcodec(
            video_path, timestamps, tolerance_s, decoder_cache=_get_cache(capacity)
        )


def clear_video_cache():
    with _lock:
        if _cache is not None:
            _cache.clear()


atexit.register(clear_video_cache)
