"""Two same-observation, same-noise prompt interventions on the saved sample."""
import gc
import hashlib
import json
import torch
from evaluate_postclose_four_chunks import OUT, infer, dump

COMMON = ('Take the soft package out of the box. If the label is facing up, keep it facing up '
          'and place the package on the roller conveyor. If the label is facing down, flip '
          'the package so the label faces up, then place it on the roller conveyor.')

if __name__=='__main__':
    manifest=json.loads((OUT/'manifest.json').read_text())
    from pathlib import Path
    assert hashlib.sha256(Path(manifest['norm']).read_bytes()).hexdigest()==manifest['norm_sha256']
    assert hashlib.sha256((Path(manifest['checkpoint'])/'lingbotvla_cli.yaml').read_bytes()).hexdigest()==manifest['config_sha256']
    for mode,prompt in [('empty',''),('conditional',COMMON)]:
        dest=OUT/'prompt_ablation'/mode
        dest.mkdir(parents=True,exist_ok=True)
        dump(dest/'manifest.json',{**manifest,'prompt_mode':mode,'prompt_override':prompt,
             'protocol':'Same baseline sample, same per-observation seeds and preprocessing; only task text changes.',
             'baseline_manifest_sha256':hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()})
        print('START',mode,flush=True)
        infer(manifest,output_dir=dest,prompt_override=prompt)
        gc.collect();torch.cuda.empty_cache()
