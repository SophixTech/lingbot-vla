"""Reuse the original 128-episode postclosure protocol with a new checkpoint."""
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
import evaluate_postclose_four_chunks as evaluation
import analyze_postclose_four_chunks as analysis

out = Path('/home/bjtc/Sophix/lingbot-vla/output/label_down_fast/postclose_four_chunks')
out.mkdir(parents=True, exist_ok=True)
source = evaluation.OUT
manifest = json.loads((source / 'manifest.json').read_text())
checkpoint = out.parent / 'checkpoints/global_step_1000/hf_ckpt'
cfg = yaml.safe_load((checkpoint / 'lingbotvla_cli.yaml').read_text())
norm = Path(cfg['data']['norm_stats_file'])
manifest.update(checkpoint=str(checkpoint), norm=str(norm),
                norm_sha256=hashlib.sha256(norm.read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256((checkpoint/'lingbotvla_cli.yaml').read_bytes()).hexdigest(),
                reference_manifest=str(source/'manifest.json'),
                limitations='Same original 64 up + 64 down episodes. Up prompt/orientation absent from this model training. Recorded observations refreshed every 50 frames; not autonomous rollout.')
training = set(json.loads(Path(cfg['data']['train_path']).joinpath('selection_report.json').read_text())['selected_episode_indices'])
manifest['training_overlap'] = {str(label):sum(r['episode'] in training for r in manifest['selected_rows'] if r['label_up']==label) for label in (True,False)}
evaluation.dump(out/'manifest.json', manifest)
with np.load(source/'ground_truth.npz') as z:
    np.savez_compressed(out/'ground_truth.npz', **{k:z[k] for k in z.files})
evaluation.OUT=out
evaluation.infer(manifest)
analysis.OUT=out
analysis.main()
