import json, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch, yaml
import matplotlib.pyplot as plt
from safetensors import safe_open
from transformers import AutoConfig
from lerobot.configs.policies import PreTrainedConfig
from deploy.lingbot_vla_policy import merge_qwen_config
from lingbotvla.models import build_processor
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
from lingbotvla.utils.lora_utils import add_lora_to_model
from lingbotvla.data.vla_data.base_dataset import VLADataset

ROOT=Path('/home/bjtc/Sophix/lingbot-vla')
CK=ROOT/'output/label_down_fast/checkpoints/global_step_1000/hf_ckpt'
OUT=ROOT/'output/label_down_fast/trajectory_eval'
PROMPT='Pick the soft package from the box, flip it so the label faces up, and place it on the conveyor belt.'

def main():
    OUT.mkdir(parents=True,exist_ok=True); cfg=yaml.safe_load((CK/'lingbotvla_cli.yaml').read_text())
    config=PreTrainedConfig.from_pretrained(str(CK)); config.__dict__.update({k:v for k,v in {**cfg['model'],**cfg['train']}.items() if not hasattr(config,k)}); config.attention_implementation='eager'; tok=os.environ['QWEN25_PATH']; config.tokenizer_path=tok; config=merge_qwen_config(config,AutoConfig.from_pretrained(tok)); config.use_cache=True
    proc=build_processor(tok); policy=LingbotVlaPolicy(config,tokenizer_path=tok); policy=add_lora_to_model(policy,lora_rank=cfg['train']['lora_rank'],lora_alpha=cfg['train']['lora_alpha'],lora_target_modules=cfg['train']['lora_target_modules'],lora_target_modules_support=('q_proj','k_proj','v_proj','o_proj'))
    weights={}
    for p in CK.glob('*.safetensors'):
      with safe_open(p,framework='pt',device='cpu') as f: weights.update({k:f.get_tensor(k) for k in f.keys()})
    policy.load_state_dict(weights,strict=True); policy.to('cuda',dtype=torch.bfloat16).eval()
    data=SimpleNamespace(**cfg['data']); ds=VLADataset(data.train_path,data.data_name,data,'configs/robot_configs',config=config,tokenizer=proc.tokenizer,image_processor=proc.image_processor)
    ep=4; meta=ds.dataset.meta.episodes[ep]; frame=int(meta['length']*0.40); idx=int(meta['dataset_from_index'])+frame; batch=ds.getdata(idx); target=ds.feature_transform.unapply({**batch,'actions':batch['actions'].clone()})['action'].numpy()
    torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    args=[batch[k].unsqueeze(0).to('cuda',dtype=torch.bfloat16 if k in ('images','state') else batch[k].dtype) for k in ('images','img_masks','lang_tokens','lang_masks','state')]
    with torch.inference_mode(): pred=policy.model.sample_actions(*args,num_steps=10).squeeze(0).float().cpu()
    actual=ds.feature_transform.unapply({**batch,'actions':pred})['action'].numpy(); n=int((~batch['action_is_pad'].bool()).sum()); n=min(n,50); recorded=target[:n,7:14]; predicted=actual[:n,7:14]; err=np.rad2deg(predicted-recorded)
    x=np.arange(1,n+1); fig,ax=plt.subplots(7,1,figsize=(12,18),sharex=True); colors=['#1677c8','#e87516']
    for j,a in enumerate(ax):
      a.plot(x,np.rad2deg(recorded[:,j]),'--',color=colors[0],label='recorded' if j==0 else None); a.plot(x,np.rad2deg(predicted[:,j]),'-',color=colors[1],label='model prediction' if j==0 else None); a.set_ylabel(f'R joint {j+1}\n(deg)'); a.grid(alpha=.25)
    ax[0].set_title(f'label_down_fast | episode {ep}, source frame {frame} | 50-frame open-loop chunk'); ax[0].legend(loc='best'); ax[-1].set_xlabel('Future frame'); fig.tight_layout(); fig.savefig(OUT/'episode_0004_frame_40pct_right_arm.png',dpi=160); plt.close(fig)
    report={'checkpoint':str(CK),'episode':ep,'source_frame':frame,'source_global_index':idx,'prompt':PROMPT,'denoising_steps':10,'valid_frames':n,'right_arm_mae_deg':float(np.abs(err).mean()),'right_arm_rmse_deg':float(np.sqrt(np.square(err).mean())),'right_arm_p95_abs_deg':float(np.quantile(np.abs(err),.95)),'per_joint_mae_deg':np.abs(err).mean(axis=0).tolist(),'mapping':'source-global index and original Parquet episode boundary'}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2)); np.savez_compressed(OUT/'trajectories.npz',recorded_deg=np.rad2deg(recorded),predicted_deg=np.rad2deg(predicted),error_deg=err)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
