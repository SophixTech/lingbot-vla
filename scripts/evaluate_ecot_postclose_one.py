"""Recorded-observation replay, four native chunks; no robot connection."""
import json, os
from pathlib import Path
from types import SimpleNamespace
import cv2, numpy as np, torch, pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.transforms.v2 import Resize
from scripts.render_heldout_ecot import CK, CFG, EP
from scripts.evaluate_label_conditioned_actions import load_policy
from lingbotvla.data.vla_data.utils import FeatureTransform

def main():
    root=Path(CFG['data']['train_path'])
    index=[json.loads(s) for s in Path(CFG['data']['ecot_annotations_path'],'annotations.jsonl').read_text().splitlines() if s.strip()]
    assert EP not in {r['episode_index'] for r in index if r['split']=='train'}
    rows=pq.read_table(root/f'data/chunk-000/file-{EP:03d}.parquet').to_pylist()
    assert all(r['episode_index']==EP and r['frame_index']==i for i,r in enumerate(rows))
    recorded=np.asarray([r['action'] for r in rows])
    closed=recorded[:,15]>.5
    crossings=np.flatnonzero(np.diff(closed.astype(int))==1)+1
    assert not closed[0] and len(crossings)
    close=int(crossings[0]); assert close+200<=len(rows)
    os.environ['QWEN25_PATH']=CFG['model']['tokenizer_path']
    policy,processor,config=load_policy(CK,CFG)
    dc=SimpleNamespace(**CFG['data'])
    for key in ('max_state_dim','max_action_dim','resize_imgs_with_padding','tokenizer_max_length'):
        setattr(dc,key,getattr(config,key))
    transform=FeatureTransform(Path(dc.robot_config_root)/f'{dc.data_name}.yaml',dc,processor.tokenizer,processor.image_processor,chunk_size=50,norm_stats_path=dc.norm_stats_file)
    predictions=[]
    for k in range(4):
        frame=close+50*k
        raw={key:torch.tensor(v) for key,v in rows[frame].items() if isinstance(v,(int,float,list))}
        for key in transform.org_features['images']:
            cap=cv2.VideoCapture(str(root/f'videos/{key}/chunk-000/file-{EP:03d}.mp4'))
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read();cap.release();assert ok
            raw[key]=Resize((224,224))(torch.from_numpy(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255)
        raw['task']='Take out the top soft package from the box; if the label is facing up, place it directly on the conveyor belt; if the label is facing down, flip it before placing it on the conveyor belt.'
        for key in transform.org_features['actions']:
            raw[key]=torch.zeros(50,16);raw[key+'_is_pad']=torch.zeros(50,dtype=torch.bool)
        item=transform.apply(raw)
        args=[item[key].unsqueeze(0).cuda().to(torch.bfloat16 if key in ('images','state') else item[key].dtype) for key in ('images','img_masks','lang_tokens','lang_masks','state')]
        torch.manual_seed(42+k)
        with torch.inference_mode(): pred=policy.model.sample_actions(*args,num_steps=10)[0].float().cpu()
        action=transform.unapply({**item,'actions':pred})['action'].numpy()
        assert action.shape==(50,16) and np.isfinite(action).all()
        predictions.append(action); print('chunk',k,flush=True)
    pred=np.concatenate(predictions);truth=recorded[close:close+200]
    error=np.rad2deg(pred[:,7:14]-truth[:,7:14])
    out=Path(__file__).resolve().parents[1]/'docs/ecot_conditioned_heldout_0010';out.mkdir(exist_ok=True)
    metrics={'episode':EP,'checkpoint':str(CK),'closure_frame':close,'protocol':'Recorded observations at closure + 0/50/100/150; 10 denoising steps; seeds 42..45; comparison to recorded actions, not physical execution.', 'mae_deg':float(abs(error).mean()),'mse_deg2':float((error**2).mean()),'per_joint_mae_deg':abs(error).mean(0).tolist()}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    np.savez(out/'trajectories.npz',prediction=pred,recorded=truth,closure_action=recorded[close])
    fig,axs=plt.subplots(4,2,figsize=(13,14),constrained_layout=True)
    for j,ax in enumerate(axs.flat):
        if j==7: ax.axis('off');continue
        anchor=recorded[close,7+j]
        ax.plot(np.arange(1,201),np.rad2deg(pred[:,7+j]-anchor),label='model',color='#1f77b4')
        ax.plot(np.arange(1,201),np.rad2deg(truth[:,7+j]-anchor),'--',label='recorded action',color='#d95f02')
        for boundary in (50,100,150):ax.axvline(boundary,color='.6',ls=':',lw=.8)
        ax.set(title=f'Right joint {j+1}',xlabel='Frame after closure (1-200)',ylabel='Change from closure (deg)');ax.grid(alpha=.25);ax.legend()
    fig.suptitle(f'ECOT conditioned | held-out episode {EP} | closure frame {close}\n4 predictions x 50 frames | recorded observations | MAE {metrics["mae_deg"]:.2f} deg')
    fig.savefig(out/'right_arm_postclose.png',dpi=160);fig.savefig(out/'right_arm_postclose.pdf')
    print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
