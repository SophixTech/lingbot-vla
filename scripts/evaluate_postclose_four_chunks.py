"""Offline 4x50-frame model/ground-truth comparison after first gripper closure.

Use an offline policy with deployment preprocessing; never import robot clients.
No filtered LeRobot dataset: retain original global indices and select jobs.
"""
import argparse
import hashlib
import json
import os
import time
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(os.environ.get('POSTCLOSE_OUTPUT_ROOT', '/home/bjtc/Sophix/lingbot-vla/output/marked_full0919'))
DATA = Path(os.environ.get('POSTCLOSE_DATA_ROOT', '/home/bjtc/Sophix/datasets/marked_data_0916'))
OUT = Path(os.environ.get('POSTCLOSE_OUT', str(ROOT / 'postclose_four_chunks_20260921')))
EXECUTE_FRAMES = int(os.environ.get('POSTCLOSE_EXECUTE_FRAMES', '50'))
EVAL_FRAMES = int(os.environ.get('POSTCLOSE_EVAL_FRAMES', '200'))
PROMPTS = {
    True: 'Pick the soft package from the box and place it on the conveyor belt, keeping the label facing up.',
    False: 'Pick the soft package from the box, flip it so the label faces up, and place it on the conveyor belt.'}


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def prepare(per_label):
    assert 1 <= EXECUTE_FRAMES <= 50 and EVAL_FRAMES == 200 and EVAL_FRAMES % EXECUTE_FRAMES == 0
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'manifest.json').exists():
        raise FileExistsError(f'Refusing to overwrite an existing evaluation: {OUT}')
    labs = {r['episode_index']: r['extra_labels']['self_annotation'] for r in
            map(json.loads, (DATA/'meta/annotations/episode_labels.jsonl').read_text().splitlines())}
    meta = sorted([r for p in (DATA/'meta/episodes').glob('**/*.parquet')
                   for r in pq.read_table(p).to_pylist()], key=lambda r:r['episode_index'])
    rows, truth, anchors, exclusions = [], [], [], []
    for m in meta:
        ep = int(m['episode_index']); label = labs[ep]
        if label['had_regrasp']:
            exclusions.append({'episode':ep,'reason':'had_regrasp','label_up':bool(label['grasp_label_up'])})
            continue
        p = DATA/f"data/chunk-{m['data/chunk_index']:03d}/file-{m['data/file_index']:03d}.parquet"
        t = pq.read_table(p).to_pydict()
        a = np.asarray(t['action'], float)
        assert np.isfinite(a).all() and len(a) == m['length']
        assert np.all(np.asarray(t['episode_index']) == ep)
        np.testing.assert_array_equal(t['frame_index'], np.arange(len(a)))
        np.testing.assert_array_equal(t['index'], np.arange(m['dataset_from_index'], m['dataset_to_index']))
        # First upward crossing of 0.5 in right-gripper COMMAND, matching prior audit.
        closed = a[:,15] > .5
        crosses = np.flatnonzero(np.diff(closed.astype(int)) == 1)+1
        if closed[0] or not len(crosses):
            exclusions.append({'episode':ep,'reason':'no bounded first closure crossing','label_up':bool(label['grasp_label_up'])})
            continue
        close = int(crosses[0])
        if close+200 > len(a):
            exclusions.append({'episode':ep,'reason':'incomplete 200-frame future'}); continue
        rows.append({'episode':ep,'label_up':bool(label['grasp_label_up']), 'closure_frame':close,
                     'closure_timestamp_s':float(t['timestamp'][close]), 'global_index':int(t['index'][close]),
                     'length':len(a), 'number_closure_crossings':len(crosses)})
        truth.append(a[close:close+200]); anchors.append(a[close])
    rng = np.random.default_rng(9921)
    groups = []
    for label in (True, False):
        pool = [i for i,r in enumerate(rows) if r['label_up']==label]
        if len(pool) < 2:
            raise ValueError('At least two eligible episodes per label are required')
        groups.append(sorted(rng.choice(pool, min(per_label,len(pool)), replace=False).tolist()))
    # Interleave groups to reduce temporal/load/PRNG scheduling effects.
    chosen = [i for pair in zip_longest(*groups) for i in pair if i is not None]
    ck = Path(os.environ.get('POSTCLOSE_CHECKPOINT', str(ROOT/'checkpoints/global_step_10000/hf_ckpt')))
    norm = Path(os.environ.get('POSTCLOSE_NORM', '/home/bjtc/Sophix/datasets/marked_full0919_stats/norm_stats.json'))
    import yaml
    cfg = yaml.safe_load((ck/'lingbotvla_cli.yaml').read_text())
    assert norm.resolve() == Path(cfg['data']['norm_stats_file']).resolve()
    heldout = None
    split_path = DATA.parent/'split_manifest.json'
    if DATA.name == 'test' and split_path.exists():
        train = Path(cfg['data']['train_path'])
        assert train.resolve() == (DATA.parent/'train').resolve()
        def mapping(path):
            return [json.loads(line) for line in (path/'episode_mapping.jsonl').read_text().splitlines()]
        tm, vm = mapping(train), mapping(DATA)
        train_ids = {r['source_episode_id'] for r in tm}
        assert not train_ids & {r['source_episode_id'] for r in vm}
        assert len(vm) == len(meta)
        lookup = {r['episode_index']:r for r in vm}
        for r in rows:
            r.update(source_episode_index=lookup[r['episode']]['source_episode_index'],source_episode_id=lookup[r['episode']]['source_episode_id'])
        provenance = json.loads((norm.parent/'provenance.json').read_text())
        assert provenance['test_episodes_used'] == 0
        assert provenance['sha256'] == hashlib.sha256(norm.read_bytes()).hexdigest()
        heldout = dict(train_episodes=len(tm),test_episodes=len(vm),source_episode_overlap=0,
                       split_manifest_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),test_episodes_used_for_norm=0)
    manifest = dict(checkpoint=str(ck),norm=str(norm),norm_sha256=hashlib.sha256(norm.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((ck/'lingbotvla_cli.yaml').read_bytes()).hexdigest(), dataset=str(DATA),
        seed_episode_selection=9921, sample_count=len(chosen), eligible_count=len(rows), exclusions=exclusions,
        eligible_rows=rows, selected_population_indices=chosen, selected_rows=[rows[i] for i in chosen],
        inference_backend='Standalone offline policy, strict checkpoint loading, bf16, eager', denoising_steps=10, rtc='off',
        image_preprocess='Deployment server: original decoded RGB uint8 -> PIL bilinear 224x224 -> checkpoint processor',
        noise='Per observation seed 992100+n_chunks*selected_index+chunk_index; n_chunks=200/execute_frames; one draw per observation',
        alignment='First right-gripper action upward crossing >0.5; proxy for closure, not physical grasp confirmation',
        windows_zero_based=[[i, i + EXECUTE_FRAMES - 1] for i in range(0, EVAL_FRAMES, EXECUTE_FRAMES)],
        execute_frames=EXECUTE_FRAMES, model_chunk_frames=50, eval_frames=EVAL_FRAMES, fps=30,
        heldout_verification=heldout,
        protocol='Non-regrasp only; real image/state at closure plus multiples of execute_frames; own annotation prompt; predict 50 frames per observation, retain first execute_frames, repeat over 200 frames. '+('Held-out test-set' if heldout else 'Training-set')+' teacher-forced offline evaluation, not closed-loop rollout.',
        relative_anchor='Same ground-truth action at closure subtracted from both predicted and real targets; no reset at chunk boundaries.')
    dump(OUT/'manifest.json',manifest)
    np.savez_compressed(OUT/'ground_truth.npz',action=np.asarray(truth),anchor=np.asarray(anchors),
                        label_up=np.asarray([r['label_up'] for r in rows]),selected=chosen)
    print('PREPARED',len(rows),'eligible,',len(chosen),'selected; exclusions',exclusions,flush=True)
    return manifest


def infer(manifest, output_dir=None, prompt_override=None):
    import torch
    import yaml
    from PIL import Image
    from types import SimpleNamespace
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from evaluate_label_conditioned_actions import load_policy, action_from_model
    ck=Path(manifest['checkpoint']);cfg=yaml.safe_load((ck/'lingbotvla_cli.yaml').read_text())
    policy,processor,config=load_policy(ck,cfg)
    dc=SimpleNamespace(**cfg['data'])
    assert Path(dc.norm_stats_file).resolve() == Path(manifest['norm']).resolve()
    for key in ('max_state_dim','max_action_dim','resize_imgs_with_padding','tokenizer_max_length'):
        setattr(dc,key,getattr(config,key))
    transform=FeatureTransform('configs/robot_configs/package_a2d.yaml',dc,processor.tokenizer,
                               processor.image_processor,norm_stats_path=dc.norm_stats_file)
    wrapper=SimpleNamespace(feature_transform=transform)
    output_dir=Path(output_dir) if output_dir is not None else OUT
    output_dir.mkdir(parents=True,exist_ok=True)
    if prompt_override is not None:
        tokens=processor.tokenizer('<bos>'+prompt_override+'\n')['input_ids']
        assert len(tokens)<=dc.tokenizer_max_length, 'Prompt would be truncated'
        dump(output_dir/'prompt_tokenization.json',dict(prompt=prompt_override,token_count=len(tokens),
             max_length=dc.tokenizer_max_length,empty_means='Empty task string; standard boundary tokens retained'))
    # Do not filter episodes; video and state global indices remain unchanged.
    ds=LeRobotDataset(manifest['dataset'],image_transforms=None,delta_timestamps=None)
    ds.hf_dataset=ds.hf_dataset.sort('index')
    truth=np.load(OUT/'ground_truth.npz');selected=truth['selected']
    n_chunks=EVAL_FRAMES//EXECUTE_FRAMES
    total=len(selected)*n_chunks;done=0;start=time.monotonic()
    samples=output_dir/'offline_predictions';samples.mkdir(exist_ok=True)
    try:
        for i,r in enumerate(manifest['selected_rows']):
            for k in range(n_chunks):
                file=samples/f"episode-{r['episode']:06d}-chunk-{k}.npz"
                if file.exists():
                    done+=1;continue
                idx=r['global_index']+EXECUTE_FRAMES*k;raw=ds[idx]
                assert int(raw['episode_index'])==r['episode']
                assert int(raw['frame_index'])==r['closure_frame']+EXECUTE_FRAMES*k
                np.testing.assert_allclose(raw['action'].numpy(),truth['action'][selected[i],EXECUTE_FRAMES*k],atol=0,rtol=0)
                obs={'observation.state':raw['observation.state'].numpy(),
                     'task':PROMPTS[r['label_up']] if prompt_override is None else prompt_override}
                for key in ds.meta.camera_keys:
                    obs[key]=(raw[key].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
                for key in ds.meta.camera_keys:
                    resized=np.asarray(Image.fromarray(obs[key]).resize((224,224),Image.Resampling.BILINEAR))
                    obs[key]=torch.from_numpy(np.array(resized.transpose(2,0,1)/255.,copy=True))
                obs['observation.state']=torch.from_numpy(obs['observation.state'].copy())
                obs['action']=torch.zeros(50,16);obs['action_is_pad']=torch.zeros(50)
                batch=transform.apply(obs)
                # Match server inference including its bf16 state before unapply.
                batch['state']=batch['state'].to(torch.bfloat16).float()
                # Verify the baseline under current code before a paired intervention.
                if prompt_override is not None and i in (0,1) and k==0:
                    from lingbotvla.data.vla_data.transform import prepare_language
                    reference=dict(batch)
                    reference['lang_tokens'],reference['lang_masks']=prepare_language(
                        processor.tokenizer,{'prompt':[PROMPTS[r['label_up']]],'state':batch['state']},dc.tokenizer_max_length)
                    check=action_from_model(policy,wrapper,reference,992100+n_chunks*i+k,10)
                    saved=np.load(OUT/'offline_predictions'/file.name)['action']
                    np.testing.assert_allclose(check,saved,atol=1e-6,rtol=1e-6)
                    dump(output_dir/f'baseline_replay_{i}.json',dict(episode=r['episode'],max_abs_difference=float(np.max(np.abs(check-saved)))))
                tick=time.monotonic()
                pred=action_from_model(policy,wrapper,batch,992100+n_chunks*i+k,10)
                infer_ms=(time.monotonic()-tick)*1000
                assert pred.shape==(50,16) and np.isfinite(pred).all()
                np.savez_compressed(file,action=pred,state=obs['observation.state'].numpy(),global_index=idx,
                                    infer_ms=infer_ms)
                done+=1
                if done%16==0:
                    elapsed=time.monotonic()-start
                    print(f'INFER {done}/{total} elapsed={elapsed:.1f}s ETA={(total-done)*elapsed/done:.1f}s',flush=True)
    finally:
        del policy
    pred=np.stack([np.concatenate([np.load(samples/f"episode-{r['episode']:06d}-chunk-{k}.npz")['action'][:EXECUTE_FRAMES]
                  for k in range(n_chunks)]) for r in manifest['selected_rows']])
    np.savez_compressed(output_dir/'paired_trajectories.npz',prediction=pred,truth=truth['action'][selected],
                        anchor=truth['anchor'][selected],label_up=truth['label_up'][selected])
    print('COMPLETE',pred.shape,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--per-label',type=int,default=64);p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    manifest=prepare(args.per_label)
    if not args.prepare_only:infer(manifest)
