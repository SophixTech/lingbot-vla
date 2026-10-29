"""Stream every frame through Qwen3-VL grounding + SAM; reduce masks across GPUs."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

LABELS = {'robot_arm', 'soft_package', 'green_table', 'conveyor'}
PROMPT = '''Locate ALL visible objects belonging to these classes:
robot_arm: robot arms including their links, joints, wrists and grippers;
soft_package: flexible bags/parcels being manipulated or resting on the work surface;
green_table: the green tabletop work surface;
conveyor: the roller conveyor work surface.
Exclude floor, walls, background equipment and table legs. Give tight bounding boxes
for each visible object; use separate boxes for disconnected visible arm parts when
occluded. Do not invent invisible objects. Return ONLY a JSON array, with each item
{"label":"robot_arm","bbox_2d":[100,200,300,400]}.
This is a format example only; determine actual boxes from the image.
Use exactly one of robot_arm, soft_package, green_table, conveyor as each label.
Coordinates must be normalized to 0..1000 relative to image width/height, not pixels.'''


def parse_boxes(text, size):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    objects = json.loads(text)
    if not isinstance(objects, list) or not objects:
        raise ValueError('Expected a nonempty object array')
    w, h = size
    result = []
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError(f'Expected an object, got {obj!r}')
        label = str(obj.get('label', '')).strip().lower().replace(' ', '_').replace('-', '_')
        label = {'robotic_arm': 'robot_arm', 'robot_arms': 'robot_arm',
                 'roller_conveyor': 'conveyor', 'conveyor_belt': 'conveyor',
                 'green_tabletop': 'green_table', 'soft_parcel': 'soft_package'}.get(label, label)
        if label not in LABELS:
            raise ValueError(f'Invalid class in model output: {obj!r}')
        # The prompt specifies normalized xyxy; a redundant format tag is optional.
        if obj.get('bbox_format', 'xyxy_1000') not in ('xyxy_1000', 'xyxy'):
            raise ValueError(f'Unsupported coordinate format in model output: {obj!r}')
        box = obj.get('bbox_2d')
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('Invalid box')
        if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1000 for x in box):
            raise ValueError('Box coordinate outside 0..1000')
        x1, y1, x2, y2 = box
        if x1 >= x2 or y1 >= y2:
            raise ValueError('Empty/reversed box')
        result.append([x1*w/1000, y1*h/1000, x2*w/1000, y2*h/1000])
    return result


def index_videos(datasets, camera, max_videos=0, max_frames=0):
    records, size = [], None
    for root in datasets:
        videos = sorted(p for p in (root/'videos').rglob('*.mp4') if camera in p.parts)
        if not videos:
            raise ValueError(f'No camera videos in {root}')
        if max_videos:
            videos = videos[:max_videos]
        for path in videos:
            cap = cv2.VideoCapture(str(path))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
            if n <= 0 or min(current) <= 0 or (size and current != size):
                raise ValueError(f'Unreadable video or inconsistent resolution: {path}')
            size = current
            records.append((str(path), min(n, max_frames) if max_frames else n))
    return records, size


def assign_videos(records, world):
    loads, jobs = [0]*world, [[] for _ in range(world)]
    for record in sorted(records, key=lambda r: -r[1]):
        rank = min(range(world), key=lambda i: loads[i])
        jobs[rank].append(record)
        loads[rank] += record[1]
    return jobs


def frame_batches(path, count, batch_size):
    cap = cv2.VideoCapture(path)
    batch = []
    try:
        for frame in range(count):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f'Failed to decode {path} frame {frame}; refusing partial output')
            batch.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            if len(batch) == batch_size:
                yield frame-len(batch)+1, batch
                batch = []
        if batch:
            yield count-len(batch), batch
    finally:
        cap.release()


class Segmenter:
    def __init__(self, args, device):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, SamModel, SamProcessor
        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(str(args.grounder), local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        self.grounder = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.grounder), local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval()
        self.sam_processor = SamProcessor.from_pretrained(str(args.sam), local_files_only=True)
        self.sam = SamModel.from_pretrained(str(args.sam), local_files_only=True).to(device).eval()

    def locate(self, images, retry=False):
        messages = [[{'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': PROMPT + ('\nCheck carefully and return valid JSON with all required fields.' if retry else '')}
        ]}] for image in images]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                   return_dict=True, return_tensors='pt', padding=True).to(self.device)
        with self.torch.inference_mode():
            ids = self.grounder.generate(**inputs, max_new_tokens=2048, do_sample=False)
        return self.processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    def __call__(self, images):
        texts = self.locate(images)
        for image, text in zip(images, texts):
            try:
                boxes = parse_boxes(text, image.size)
            except (ValueError, TypeError, KeyError, AttributeError):
                retry_text = self.locate([image], retry=True)[0]
                try:
                    boxes = parse_boxes(retry_text, image.size)
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    raise ValueError(f'Grounding validation failed. Initial output: {text!r}; '
                                     f'retry output: {retry_text!r}') from exc
            # SAM uses the original image coordinates, not Qwen's normalized coordinates.
            inputs = self.sam_processor(images=image, input_boxes=[boxes], return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                outputs = self.sam(**inputs, multimask_output=True)
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs['original_sizes'].cpu(),
                inputs['reshaped_input_sizes'].cpu())[0].numpy()
            scores = outputs.iou_scores[0].float().cpu().numpy()
            keep = np.zeros((image.height, image.width), bool)
            for box, candidates, quality in zip(boxes, masks, scores):
                # Limit SAM leakage outside the grounded object, with a 4-pixel tolerance.
                x1, y1, x2, y2 = box
                x1, y1 = max(0, math.floor(x1)-4), max(0, math.floor(y1)-4)
                x2, y2 = min(image.width, math.ceil(x2)+4), min(image.height, math.ceil(y2)+4)
                chosen = candidates[int(np.argmax(quality))]
                keep[y1:y2, x1:x2] |= chosen[y1:y2, x1:x2]
            if not keep.any():
                raise RuntimeError('SAM returned an empty frame mask; refusing silent omission')
            yield keep


def save_result(output, mask, records, datasets):
    # A single contact sheet shows original and masked frames from each dataset.
    previews = []
    for root in datasets:
        subset = [r for r in records if Path(r[0]).is_relative_to(root)]
        path, n = subset[len(subset)//2]
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n//2)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f'Cannot read preview: {path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        previews.append((root.name, rgb, np.where(mask[..., None], rgb, 0).astype(np.uint8)))
    w, h = mask.shape[1], mask.shape[0]
    sheet = Image.new('RGB', (2*w, len(previews)*(h+28)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for row, (label, raw, masked) in enumerate(previews):
        y = row*(h+28)
        draw.text((8, y+5), label+' | original / masked', fill='white')
        sheet.paste(Image.fromarray(raw), (0, y+28))
        sheet.paste(Image.fromarray(masked), (w, y+28))
    sheet.save(output/'masked_preview.jpg', quality=92)
    Image.fromarray(mask.astype(np.uint8)*255).save(output/'mask.png')


def main():
    import torch
    import torch.distributed as dist
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', type=Path, required=True)
    p.add_argument('--grounder', type=Path, required=True)
    p.add_argument('--sam', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory; refuses overwrite')
    p.add_argument('--camera', default='observation.images.cam_high_rgb')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-videos-per-dataset', type=int, default=0, help='Smoke test only; 0 = all')
    p.add_argument('--max-frames-per-video', type=int, default=0, help='Smoke test only; 0 = all')
    args = p.parse_args()
    if min(args.batch_size, args.threads) < 1 or min(args.max_videos_per_dataset, args.max_frames_per_video) < 0:
        p.error('Invalid batch/thread/limit argument')
    args.datasets = [d.resolve() for d in args.datasets]
    if len(set(args.datasets)) != len(args.datasets):
        p.error('Duplicate dataset roots')
    if not torch.cuda.is_available():
        p.error('CUDA GPU required')
    rank, world, local = int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1)), int(os.getenv('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    if world > 1:
        # CPU reduction only at completion; workers may finish at substantially different times.
        dist.init_process_group('gloo', timeout=timedelta(days=7))
    payload = [None]
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        payload[0] = index_videos(args.datasets, args.camera, args.max_videos_per_dataset, args.max_frames_per_video)
    if world > 1:
        dist.broadcast_object_list(payload, src=0)
    records, (w, h) = payload[0]
    jobs = assign_videos(records, world)[rank]
    expected = sum(n for _, n in jobs)
    print(f'Rank {rank}: {len(jobs)} videos, {expected} frames; global {sum(n for _, n in records)}', flush=True)
    segmenter = Segmenter(args, f'cuda:{local}') if jobs else None
    mask = np.zeros((h, w), bool)
    done, start = 0, time.monotonic()
    last_progress = start - 10
    for path, count in jobs:
        for first, images in frame_batches(path, count, args.batch_size):
            try:
                for frame_mask in segmenter(images):
                    mask |= frame_mask
                    done += 1
            except Exception as exc:
                raise RuntimeError(f'{path}: batch starting at frame {first}') from exc
            now = time.monotonic()
            if now-last_progress >= 10 or done == expected:
                fraction = done/expected
                filled = int(30*fraction)
                bar = '='*filled + '-'*(30-filled)
                speed = done/max(now-start, 1e-6)
                print(f'GPU {local} [{bar}] {fraction:6.2%} | {done:,}/{expected:,} frames | '
                      f'{speed:.2f} fps | ETA {(expected-done)/speed/3600:.2f}h | '
                      f'keep {mask.mean():.2%}', flush=True)
                last_progress = now
    assert done == expected
    merged = torch.from_numpy(mask.astype(np.uint8))
    counts = torch.tensor(done, dtype=torch.int64)
    if world > 1:
        dist.reduce(merged, dst=0, op=dist.ReduceOp.MAX)
        dist.reduce(counts, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        assert counts.item() == sum(n for _, n in records)
        save_result(args.output, merged.numpy().astype(bool), records, args.datasets)
        mode = 'SMOKE TEST' if args.max_videos_per_dataset or args.max_frames_per_video else 'ALL FRAMES'
        print(f'{mode} COMPLETE: {counts.item()} frames, keep {merged.float().mean().item():.2%}; {args.output}', flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    # Include the worker traceback in torchrun's final Root Cause report.
    from torch.distributed.elastic.multiprocessing.errors import record
    record(main)()
