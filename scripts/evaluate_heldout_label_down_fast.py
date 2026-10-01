"""Evaluate label_down_fast on every eligible label-down episode outside its train set."""
from pathlib import Path
import hashlib, json, os
import numpy as np
import pyarrow.parquet as pq

DATA=Path('/home/bjtc/Sophix/datasets/marked_data_0916')
TRAIN_VIEW=Path('/home/bjtc/Sophix/datasets/label_down_fast_150')
ROOT=Path('/home/bjtc/Sophix/lingbot-vla/output/label_down_fast')
OUT=ROOT/'heldout_label_down_postclose'
CK=ROOT/'checkpoints/global_step_1000/hf_ckpt'
N=200

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        raise RuntimeError(f'Output already exists: {OUT}; refusing overwrite')
    train_ids=set(json.loads((TRAIN_VIEW/'selection_report.json').read_text())['selected_episode_indices'])
    labels={int(r['episode_index']):r['extra_labels']['self_annotation'] for r in map(json.loads,(DATA/'meta/annotations/episode_labels.jsonl').read_text().splitlines())}
    meta=sorted([r for p in (DATA/'meta/episodes').glob('**/*.parquet') for r in pq.read_table(p).to_pylist()],key=lambda r:r['episode_index'])
    rows=[]; truth=[]; anchors=[]; excluded=[]
    for m in meta:
        ep=int(m['episode_index']); lab=labels[ep]
        if lab['had_regrasp'] or lab['grasp_label_up'] is not False:
            continue
        if ep in train_ids:
            excluded.append({'episode':ep,'reason':'train_episode'}); continue
        p=DATA/f"data/chunk-{m['data/chunk_index']:03d}/file-{m['data/file_index']:03d}.parquet"
        t=pq.read_table(p).to_pydict(); a=np.asarray(t['action'],float)
        assert len(a)==m['length'] and np.isfinite(a).all()
        np.testing.assert_array_equal(t['episode_index'],np.full(len(a),ep))
        np.testing.assert_array_equal(t['frame_index'],np.arange(len(a)))
        np.testing.assert_array_equal(t['index'],np.arange(m['dataset_from_index'],m['dataset_to_index']))
        closed=a[:,15]>.5; crosses=np.flatnonzero((closed[1:]) & (~closed[:-1]))+1
        if closed[0] or len(crosses)==0 or int(crosses[0])+N>len(a):
            excluded.append({'episode':ep,'reason':'no_complete_200_frame_postclosure'}); continue
        close=int(crosses[0]); rows.append({'episode':ep,'label_up':False,'closure_frame':close,'closure_timestamp_s':float(t['timestamp'][close]),'global_index':int(t['index'][close]),'length':len(a),'number_closure_crossings':len(crosses)})
        truth.append(a[close:close+N]); anchors.append(a[close])
    manifest={'checkpoint':str(CK),'norm':str(DATA.parent/'label_down_fast_stats/norm_stats.json'),'dataset':str(DATA),'selected_rows':rows,'selected_population_indices':list(range(len(rows))),'eligible_count':len(rows),'sample_count':len(rows),'excluded_count':len(excluded),'excluded_summary':excluded,'train_episode_count':len(train_ids),'train_episode_overlap':0,'execute_frames':50,'model_chunk_frames':50,'eval_frames':N,'denoising_steps':10,'protocol':'All eligible label-down non-regrasp episodes outside the 150-episode training whitelist; source-global indices retained; real observations refreshed every 50 frames; recorded action targets compared with model predictions.','norm_sha256':hashlib.sha256((DATA.parent/'label_down_fast_stats/norm_stats.json').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((CK/'lingbotvla_cli.yaml').read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)); np.savez_compressed(OUT/'ground_truth.npz',action=np.asarray(truth),anchor=np.asarray(anchors),label_up=np.zeros(len(rows),dtype=bool),selected=np.arange(len(rows)))
    print(json.dumps({'eligible_test_episodes':len(rows),'train_whitelist':len(train_ids),'excluded':len(excluded),'frame_count':sum(r['length'] for r in rows)},indent=2),flush=True)
    # Reuse the original strict model loader/inference loop.
    import evaluate_postclose_four_chunks as ev
    ev.OUT=OUT
    ev.infer(manifest)
    # Plot all down test trajectories: model solid, recorded dashed.
    z=np.load(OUT/'paired_trajectories.npz'); anchor=np.rad2deg(z['anchor'][:,None,7:14]); pred=np.rad2deg(z['prediction'][:,:,7:14])-anchor; rec=np.rad2deg(z['truth'][:,:,7:14])-anchor; x=np.arange(1,N+1)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True); axes=axes.ravel()
    for j,ax in enumerate(axes[:7]):
        ax.plot(x,pred[:,:,j].mean(0),color='#d95f02',lw=1.7,label='down model')
        ax.plot(x,rec[:,:,j].mean(0),color='#d95f02',ls='--',lw=1.7,label='down recorded')
        for b in (50.5,100.5,150.5):ax.axvline(b,color='.6',lw=.7,ls=':')
        ax.set_title(f'Right joint {j+1}'); ax.set_ylabel('Change from closure (deg)'); ax.set_xlabel('Frame after closure (1-200)'); ax.grid(alpha=.2)
    axes[0].legend(frameon=False); axes[-1].axis('off'); fig.suptitle(f'All held-out label-down episodes ({len(rows)} episodes) | Four predictions with recorded observations'); fig.tight_layout(); fig.savefig(OUT/'heldout_label_down_model_vs_recorded.png',dpi=160); plt.close(fig)
    e=pred-rec; report=json.loads((OUT/'manifest.json').read_text()); report.update({'metrics':{'episodes':len(rows),'mae_deg':float(np.abs(e).mean()),'rmse_deg':float(np.sqrt(np.square(e).mean())),'p95_abs_deg':float(np.quantile(np.abs(e),.95)),'mae_deg_by_window':np.abs(e).reshape(len(rows),4,50,7).mean((0,2,3)).tolist(),'rmse_deg_by_window':np.sqrt(np.square(e).reshape(len(rows),4,50,7).mean((0,2,3))).tolist(),'mae_deg_by_joint':np.abs(e).mean((0,1)).tolist()},'mapping_check':'PASS: source episode and global index ranges verified before inference'})
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report['metrics'],indent=2),flush=True)
if __name__=='__main__':main()
