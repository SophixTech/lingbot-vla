"""Reproducible in-dataset open-loop action error, no robot connection."""
import hashlib
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from safetensors import safe_open
from transformers import AutoConfig
from lerobot.configs.policies import PreTrainedConfig
from deploy.lingbot_vla_policy import merge_qwen_config, set_seed_everywhere
from lingbotvla.models import build_processor
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
from lingbotvla.utils.lora_utils import add_lora_to_model
from lingbotvla.data.vla_data.base_dataset import VLADataset

ROOT=Path('/home/bjtc/Sophix/lingbot-vla/output/marked_full0919')
CK=ROOT/'checkpoints/global_step_10000/hf_ckpt'
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,default=ROOT/'evaluation_mse_20260921')
parser.add_argument('--checkpoint',type=Path,default=CK)
parser.add_argument('--denoising-steps',type=int,default=10)
parser.add_argument('--evaluation-data',type=str,default=None)
parser.add_argument('--prompt-mode',choices=['annotated','generic'],default='annotated')
options=parser.parse_args()
OUT=options.output_dir
CK=options.checkpoint
OUT.mkdir(exist_ok=True)
cfg=yaml.safe_load((CK/'lingbotvla_cli.yaml').read_text())
set_seed_everywhere(42)
config=PreTrainedConfig.from_pretrained(str(CK))
config.__dict__.update({k:v for k,v in {**cfg['model'],**cfg['train']}.items() if not hasattr(config,k)})
config.attention_implementation='eager'
config.tokenizer_path=os.environ['QWEN25_PATH']
config=merge_qwen_config(config,AutoConfig.from_pretrained(config.tokenizer_path))
config.use_cache=True
processor=build_processor(config.tokenizer_path)
print('Loading final checkpoint on CPU',flush=True)
policy=LingbotVlaPolicy(config,tokenizer_path=config.tokenizer_path)
policy=add_lora_to_model(policy,lora_rank=cfg['train']['lora_rank'],lora_alpha=cfg['train']['lora_alpha'],
    lora_target_modules=cfg['train']['lora_target_modules'],lora_target_modules_support=('q_proj','k_proj','v_proj','o_proj'))
weights={}
for p in CK.glob('*.safetensors'):
    with safe_open(p,framework='pt',device='cpu') as f:
        weights.update({k:f.get_tensor(k) for k in f.keys()})
policy.load_state_dict(weights,strict=True)
del weights
# Convert while transferring to avoid a transient full-FP32 GPU allocation.
policy=policy.to(device='cuda',dtype=torch.bfloat16).eval()
data=SimpleNamespace(**cfg['data'])
if options.evaluation_data is not None:
    data.train_path=options.evaluation_data
    data.exclude_regrasp=False
ds=VLADataset(data.train_path,data.data_name,data,'configs/robot_configs',config=config,
    tokenizer=processor.tokenizer,image_processor=processor.image_processor)
assert ds.selected_episodes is None and len(ds)==748377
groups={}
for e,r in ds.episode_labels.items():
    if e in (806,865):continue
    l=r['extra_labels']['self_annotation'];key=(bool(l['had_regrasp']),bool(l['grasp_label_up']))
    groups.setdefault(key,[]).append(e)
sizes={k:len(v) for k,v in groups.items()};quotas={k:int(64*n/1062) for k,n in sizes.items()}
for k in sorted(groups,key=lambda k:64*sizes[k]/1062-quotas[k],reverse=True)[:64-sum(quotas.values())]:quotas[k]+=1
rng=np.random.default_rng(42)
selected=sorted(int(e) for k in sorted(groups) for e in rng.choice(sorted(groups[k]),quotas[k],replace=False))
jobs=[]
for e in selected+[806,865]:
    meta=ds.dataset.meta.episodes[e];length=meta['length']
    for frac in (.2,.5,.8):
        frame=int((length-1)*frac);jobs.append((e,frame,int(meta['dataset_from_index'])+frame))
manifest={'checkpoint':str(CK),'norm':data.norm_stats_file,'norm_sha256':hashlib.sha256(Path(data.norm_stats_file).read_bytes()).hexdigest(),
    'seed':42,'num_denoising_steps':options.denoising_steps,'rtc':'off','precision':'bfloat16','dataset':data.train_path,
    'prompt_mode':options.prompt_mode,
    'protocol':'In-training-dataset stratified episode sample, frames at 20/50/80 percent, all valid action horizons; not held-out validation. Groups weighted by episode counts, not by source frame counts.',
    'strata':[{'regrasp':k[0],'label_up':k[1],'population':sizes[k],'selected':quotas[k]} for k in sorted(groups)],
    'selected_episodes':selected,'additional_stress_episodes':[806,865],'jobs':jobs}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
acc={};records=[];joint_errors={}
def accumulate(group,nerr,err):
    for name,value in [('normalized_16d',nerr),('arm_14d_rad',err[:,:14]),('right_arm_7d_rad',err[:,7:14]),
                       ('left_arm_7d_rad',err[:,:7]),('gripper_2d_unit',err[:,14:]),('action_16d_mixed_units',err)]:
        d=acc.setdefault(group,{}).setdefault(name,{'squared_sum':0.,'absolute_sum':0.,'count':0})
        d['squared_sum']+=float(np.square(value).sum());d['absolute_sum']+=float(abs(value).sum());d['count']+=value.size
with torch.inference_mode(), (OUT/'samples.jsonl').open('w') as log:
    for job_id,(ep,frame,idx) in enumerate(jobs):
        if options.prompt_mode=='generic':
            raw=ds.dataset[idx]
            raw['task']='Pick a soft package from the box, orient its label upward, and place it on the conveyor belt.'
            batch=ds.feature_transform.apply(raw)
        else:
            batch=ds.getdata(idx)
        target=batch['actions'].clone();mask=batch['joint_mask'];valid=~batch['action_is_pad'].bool()
        assert int(mask.sum())==16
        torch.manual_seed(42+job_id);torch.cuda.manual_seed_all(42+job_id)
        args=[batch[k].unsqueeze(0).to(device='cuda',dtype=torch.bfloat16 if k in ('images','state') else batch[k].dtype)
              for k in ('images','img_masks','lang_tokens','lang_masks','state')]
        pred=policy.model.sample_actions(*args,num_steps=options.denoising_steps).squeeze(0).float().cpu()
        nerr=(pred[valid][:,mask]-target[valid][:,mask]).double().numpy()
        gt=ds.feature_transform.unapply({**batch,'actions':target.clone()})['action'].numpy()
        actual=ds.feature_transform.unapply({**batch,'actions':pred.clone()})['action'].numpy()
        np.testing.assert_allclose(gt[0],ds.dataset.hf_dataset[idx]['action'].numpy(),atol=2e-5,rtol=1e-5)
        err=(actual[valid.numpy()].astype(np.float64)-gt[valid.numpy()])
        assert np.isfinite(err).all() and np.isfinite(nerr).all()
        group='regular_64_episodes' if ep not in (806,865) else 'stress_806_865'
        joint_errors.setdefault(group,[]).append(err[:,7:14])
        accumulate(group,nerr,err);accumulate('all_sampled',nerr,err)
        lab=ds.episode_labels[ep]['extra_labels']['self_annotation']
        if ep not in (806,865):accumulate(f"regrasp_{int(lab['had_regrasp'])}_label_up_{int(lab['grasp_label_up'])}",nerr,err)
        r={'episode':ep,'frame':frame,'valid_horizon':int(valid.sum()),'normalized_mse':float(np.square(nerr).mean()),
           'arm_mse_rad2':float(np.square(err[:,:14]).mean()),'arm_mae_rad':float(abs(err[:,:14]).mean()),'group':group}
        log.write(json.dumps(r)+'\n');log.flush();records.append(r)
        if job_id%12==0:print('evaluated',job_id+1,'/',len(jobs),r,flush=True)
results={g:{k:{'mse':d['squared_sum']/d['count'],'mae':d['absolute_sum']/d['count'],'rmse':(d['squared_sum']/d['count'])**.5,'scalar_count':d['count']} for k,d in v.items()} for g,v in acc.items()}
(OUT/'metrics.json').write_text(json.dumps({'samples':len(records),'results':results},indent=2))
per_joint={}
for group,chunks in joint_errors.items():
    errors=np.concatenate(chunks)
    per_joint[group]=[{'joint':j+1,'valid_predictions':len(errors),
        'mae_rad':float(np.abs(errors[:,j]).mean()),
        'mae_deg':float(np.rad2deg(np.abs(errors[:,j]).mean())),
        'rmse_deg':float(np.rad2deg(np.sqrt(np.square(errors[:,j]).mean()))),
        'p95_abs_deg':float(np.rad2deg(np.quantile(np.abs(errors[:,j]),.95)))} for j in range(7)]
    np.testing.assert_allclose(np.abs(errors).mean(),results[group]['right_arm_7d_rad']['mae'])
    np.savez_compressed(OUT/f'{group}_right_arm_errors.npz',errors_rad=errors)
(OUT/'right_arm_per_joint.json').write_text(json.dumps(per_joint,indent=2))
print(json.dumps(per_joint,indent=2),flush=True)
print(json.dumps(results,indent=2),flush=True)
