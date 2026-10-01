import json
from pathlib import Path
import numpy as np

root=Path('/home/bjtc/Sophix/lingbot-vla/output/marked_full0919')
runs={'marked_full0919_10':'evaluation_right_arm_per_joint',
      **{f'interpolation_{n}':f'error_diagnosis/interpolation{n}_common_inputs' for n in [5,10,20]}}
result={}
reference=json.loads((root/'evaluation_right_arm_per_joint/manifest.json').read_text())
for name,folder in runs.items():
    p=root/folder
    manifest=json.loads((p/'manifest.json').read_text())
    assert manifest['jobs']==reference['jobs'] and manifest['seed']==reference['seed']
    samples=[json.loads(l) for l in (p/'samples.jsonl').read_text().splitlines()]
    regular=[s for s in samples if s['group']=='regular_64_episodes']
    errors=np.load(p/'regular_64_episodes_right_arm_errors.npz')['errors_rad']
    h=np.concatenate([np.arange(s['valid_horizon']) for s in regular]);assert len(h)==len(errors)
    metrics=json.loads((p/'metrics.json').read_text())['results']['regular_64_episodes']
    a=metrics['right_arm_7d_rad']
    result[name]={'right_arm_mse_rad2':a['mse'],'right_arm_mae_deg':float(np.rad2deg(a['mae'])),
        'right_arm_rmse_deg':float(np.rad2deg(a['rmse'])),
        'first_action_mae_deg':float(np.rad2deg(abs(errors[h==0]).mean())),
        'last25_actions_mae_deg':float(np.rad2deg(abs(errors[h>=25]).mean())),
        'per_joint_mae_deg':np.rad2deg(abs(errors).mean(0)).tolist(),
        'normalized_mse':metrics['normalized_16d']['mse'], 'prompt_mode':manifest.get('prompt_mode','annotated')}
report={'protocol':'Same marked dataset images/states and 64-episode/192-origin sample. Own checkpoint norms and training prompts; 5/10/20 inference steps as named. Excludes episodes 806/865. Not original-dataset evaluation or historical training loss.', 'results':result}
(root/'error_diagnosis/interpolation_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
