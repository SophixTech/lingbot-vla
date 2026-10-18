import json, os
from pathlib import Path
import cv2, torch, yaml
import pyarrow.parquet as pq
from types import SimpleNamespace
from scripts.evaluate_label_conditioned_actions import load_policy
from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
from lingbotvla.data.vla_data.utils import FeatureTransform
from torchvision.transforms.v2 import Resize

CK = Path('/var/tmp/bjtc_ecot1003_conditioned_20261009/checkpoints/global_step_10000/hf_ckpt')
CFG = yaml.safe_load((CK/'lingbotvla_cli.yaml').read_text())
EP = 10  # not present in the 609-example training index
VIDEO = Path(f'/home/bjtc/Sophix/datasets/marked_9_1_0922/train/videos/observation.images.cam_high_rgb/chunk-000/file-{EP:03d}.mp4')
OUT = Path('/home/bjtc/Sophix/worktrees/branch-ecot/heldout_episode_0010_conditioned_predictions.mp4')
NAMES = ['pick_soft_package','place_label_up','flip_and_place_label_up','retract_or_idle']

def main():
    os.environ['QWEN25_PATH'] = '/home/bjtc/Sophix/models/Qwen2.5-VL-3B-Instruct-tokenizer'
    training = [json.loads(x) for x in Path(CFG['data']['ecot_annotations_path'],'annotations.jsonl').read_text().splitlines() if x.strip()]
    assert EP not in {int(x['episode_index']) for x in training if x['split']=='train'}
    policy, processor, config = load_policy(CK, CFG)
    label_outputs = []
    hook = policy.model.ecot_label_head.register_forward_hook(
        lambda module, args, output: label_outputs.append(output.detach().float().softmax(-1).cpu()))
    dc = SimpleNamespace(**CFG['data'])
    dc.ecot_annotations_path=None; dc.ecot_annotations_only=False; dc.exclude_regrasp=False
    for key in ['max_state_dim','max_action_dim','resize_imgs_with_padding','tokenizer_max_length']:
        setattr(dc,key,getattr(config,key))
    transform=FeatureTransform(Path(dc.robot_config_root)/f'{dc.data_name}.yaml',dc,
        processor.tokenizer,processor.image_processor,chunk_size=CFG['train']['chunk_size'],norm_stats_path=dc.norm_stats_file)
    rows=pq.read_table(Path(dc.train_path)/f'data/chunk-000/file-{EP:03d}.parquet').to_pylist()
    assert all(int(r['episode_index'])==EP for r in rows)
    indices = list(range(len(rows)))
    cameras={key:cv2.VideoCapture(str(Path(dc.train_path)/f'videos/{key}/chunk-000/file-{EP:03d}.mp4')) for key in transform.org_features['images']}
    predictions=[]
    cap = cv2.VideoCapture(str(VIDEO)); fps=cap.get(cv2.CAP_PROP_FPS)
    w,h=int(cap.get(3)),int(cap.get(4)); OUT.parent.mkdir(parents=True,exist_ok=True)
    enc=cv2.VideoWriter(str(OUT),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    last=None; i=0
    with torch.inference_mode():
        while True:
            ok,frame=cap.read()
            if not ok: break
            if True:
                raw={k:torch.tensor(v) for k,v in rows[i].items() if isinstance(v,(int,float,list))}
                for key,camera in cameras.items():
                    ok,img=camera.read(); assert ok
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    raw[key]=Resize((224,224))(torch.from_numpy(img).permute(2,0,1).float()/255.)
                assert int(raw['frame_index'])==i and int(raw['episode_index'])==EP
                raw['task']='Take out the top soft package from the box; if the label is facing up, place it directly on the conveyor belt; if the label is facing down, flip it before placing it on the conveyor belt.'
                for key in transform.org_features['actions']:
                    raw[key]=torch.zeros(CFG['train']['chunk_size'],raw[key].shape[-1])
                    raw[key+'_is_pad']=torch.zeros(CFG['train']['chunk_size'],dtype=torch.bool)
                item=transform.apply(raw)
                images=item['images'].unsqueeze(0).cuda().bfloat16()
                masks=item['img_masks'].unsqueeze(0).cuda()
                toks=item['lang_tokens'].unsqueeze(0).cuda(); lm=item['lang_masks'].unsqueeze(0).cuda()
                if getattr(config, 'ecot_visibility', False):
                    label_outputs.clear()
                    # Capture the independent label head on the identical prefix.
                    def label_hook(module, args, output):
                        policy.model.ecot_label_head(args[0].to(policy.model.ecot_label_head[1].weight.dtype))
                    prefix_hook = policy.model.ecot_subtask_head.register_forward_hook(label_hook)
                    logits,bbox,visible=policy.model.predict_ecot(images,masks,toks,lm,return_visibility=True)
                    prefix_hook.remove()
                    label_prob=label_outputs[-1][0].tolist()
                    box=policy.model.nullable_bboxes(bbox,visible)[0]
                else:
                    logits,bbox=policy.model.predict_ecot(images,masks,toks,lm)
                    box=bbox[0].cpu().tolist()
                last=(int(logits.argmax(-1)[0]),box)
                label_name='up' if label_prob[1]>=label_prob[0] else 'down'
                predictions.append({'frame':i,'subtask':NAMES[last[0]],'bbox_normalized':last[1],
                    'label':label_name,'label_probabilities_down_up':label_prob,'visibility_probability':float(visible[0])})
                if i==0:
                    actions=policy.model.sample_actions(images,masks,toks,lm,item['state'].unsqueeze(0).cuda().bfloat16(),num_steps=10)
                    assert torch.isfinite(actions).all()
                    print('action_shape',list(actions.shape),flush=True)
                if i%100==0: print('frame',i,flush=True)
            if last:
                sub,b=last
                if b is not None:
                    x1,y1,x2,y2=[int(v*s) for v,s in zip(b,(w,h,w,h))]
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(frame,(0,0),(w,76),(20,20,20),-1)
                cv2.putText(frame,f'HELDOUT ep {EP} | {NAMES[sub]} | bbox {"null" if b is None else "pred"}', (8,30),0,.62,(0,255,255),2)
                cv2.putText(frame,f'frame {i} | label: {label_name} ({max(label_prob):.2f}) | visibility: {float(visible[0]):.2f}',(8,60),0,.60,(255,255,255),1)
            enc.write(frame); i+=1
    assert i==len(indices)
    cap.release(); enc.release()
    hook.remove()
    for camera in cameras.values(): camera.release()
    OUT.with_suffix('.json').write_text(json.dumps(predictions))
    print(OUT, i)
if __name__=='__main__': main()
