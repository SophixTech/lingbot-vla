import os,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch,yaml,matplotlib.pyplot as plt
from safetensors import safe_open
from transformers import AutoConfig
from lerobot.configs.policies import PreTrainedConfig
from deploy.lingbot_vla_policy import merge_qwen_config
from lingbotvla.models import build_processor
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
from lingbotvla.utils.lora_utils import add_lora_to_model
from lingbotvla.data.vla_data.base_dataset import VLADataset

ROOT=Path('/home/bjtc/Sophix/lingbot-vla'); OUT=ROOT/'output/label_down_fast/trajectory_eval_comparison'; OUT.mkdir(parents=True,exist_ok=True)
PROMPT='Pick the soft package from the box, flip it so the label faces up, and place it on the conveyor belt.'
MODELS={
 'label_down_fast':ROOT/'output/label_down_fast/checkpoints/global_step_1000/hf_ckpt',
 'marked_full0919':ROOT/'output/marked_full0919/checkpoints/global_step_10000/hf_ckpt',
 'interpolation_vla':ROOT/'output/interpolation_vla/checkpoints/global_step_10000/hf_ckpt',
}
def load(ck):
 cfg=yaml.safe_load((ck/'lingbotvla_cli.yaml').read_text()); c=PreTrainedConfig.from_pretrained(str(ck)); c.__dict__.update({k:v for k,v in {**cfg['model'],**cfg['train']}.items() if not hasattr(c,k)}); c.attention_implementation='eager'; tok=os.environ['QWEN25_PATH']; c.tokenizer_path=tok; c=merge_qwen_config(c,AutoConfig.from_pretrained(tok)); c.use_cache=True; p=build_processor(tok); m=LingbotVlaPolicy(c,tokenizer_path=tok); m=add_lora_to_model(m,lora_rank=cfg['train']['lora_rank'],lora_alpha=cfg['train']['lora_alpha'],lora_target_modules=cfg['train']['lora_target_modules'],lora_target_modules_support=('q_proj','k_proj','v_proj','o_proj')); w={}
 for f in ck.glob('*.safetensors'):
  with safe_open(f,framework='pt',device='cpu') as z:w.update({k:z.get_tensor(k) for k in z.keys()})
 m.load_state_dict(w,strict=True); return m.to('cuda',dtype=torch.bfloat16).eval(),p,cfg
def main():
 base_cfg=yaml.safe_load((MODELS['label_down_fast']/'lingbotvla_cli.yaml').read_text()); data=SimpleNamespace(**base_cfg['data']); data.exclude_regrasp=False; policy0,proc0,_=load(MODELS['label_down_fast']); ds=VLADataset(data.train_path,data.data_name,data,'configs/robot_configs',config=policy0.config,tokenizer=proc0.tokenizer,image_processor=proc0.image_processor); ep=4; meta=ds.dataset.meta.episodes[ep]; frame=int(meta['length']*.4); idx=int(meta['dataset_from_index'])+frame; batch=ds.getdata(idx); recorded=ds.feature_transform.unapply({**batch,'actions':batch['actions'].clone()})['action'].numpy(); n=min(int((~batch['action_is_pad'].bool()).sum()),50); recorded=np.rad2deg(recorded[:n,7:14]); preds={}
 for name,ck in MODELS.items():
  model,proc,cfg=load(ck); d=SimpleNamespace(**cfg['data']); d.exclude_regrasp=False; local=VLADataset(d.train_path,d.data_name,d,'configs/robot_configs',config=model.config,tokenizer=proc.tokenizer,image_processor=proc.image_processor); b=local.getdata(idx) if str(d.train_path)==str(data.train_path) else local.getdata(int(local.dataset.meta.episodes[ep]['dataset_from_index'])+frame); torch.manual_seed(2026);torch.cuda.manual_seed_all(2026); args=[b[k].unsqueeze(0).to('cuda',dtype=torch.bfloat16 if k in ('images','state') else b[k].dtype) for k in ('images','img_masks','lang_tokens','lang_masks','state')];
  with torch.inference_mode(): out=model.model.sample_actions(*args,num_steps=10).squeeze(0).float().cpu(); actual=local.feature_transform.unapply({**b,'actions':out})['action'].numpy(); preds[name]=np.rad2deg(actual[:n,7:14]); del model
 x=np.arange(1,n+1); fig,ax=plt.subplots(7,1,figsize=(13,19),sharex=True); cols={'recorded':'#333333','label_down_fast':'#d62728','marked_full0919':'#1f77b4','interpolation_vla':'#2ca02c'}
 for j,a in enumerate(ax):
  a.plot(x,recorded[:,j],'--',color=cols['recorded'],label='recorded' if j==0 else None)
  for k in MODELS:a.plot(x,preds[k][:,j],color=cols[k],label=k if j==0 else None)
  a.set_ylabel(f'R joint {j+1}\n(deg)');a.grid(alpha=.25)
 ax[0].set_title(f'Same label-down trajectory | episode {ep}, source frame {frame} | recorded vs 3 models');ax[0].legend(ncol=4);ax[-1].set_xlabel('Future frame');fig.tight_layout();fig.savefig(OUT/'episode_0004_frame_40pct_three_models.png',dpi=160);plt.close(fig)
 metrics={}
 for k,v in preds.items():
  e=v-recorded; metrics[k]={'mae_deg':float(np.abs(e).mean()),'rmse_deg':float(np.sqrt(np.square(e).mean())),'p95_abs_deg':float(np.quantile(np.abs(e),.95)),'mean_step_jitter_deg':float(np.abs(np.diff(v,axis=0)).mean()),'step_rmse_deg':float(np.sqrt(np.square(np.diff(v,axis=0)).mean())),'per_joint_mae_deg':np.abs(e).mean(axis=0).tolist()}
 report={'prompt':PROMPT,'episode':ep,'source_frame':frame,'valid_frames':n,'normalization':{k:str(yaml.safe_load((ck/'lingbotvla_cli.yaml').read_text())['data']['norm_stats_file']) for k,ck in MODELS.items()},'metrics':metrics}
 (OUT/'metrics.json').write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/'trajectories.npz',recorded=recorded,**preds);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
