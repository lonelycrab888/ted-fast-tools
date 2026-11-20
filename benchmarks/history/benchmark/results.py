"""Gather measured wall times for the 48-structure development benchmark."""
import json
from pathlib import Path
p = Path(__file__).resolve().parent
times = {}
for stage in ('merizo','chainsaw','unidoc','consensus'):
    raw = (p/'reference'/f'{stage}.time').read_text().strip()
    if not raw and stage == 'consensus':
        raw = (p/'reference'/'consensus.log').read_text().splitlines()[-1]
    times[stage] = float(raw)
reference = sum(times.values())
output = {'baseline': {'seconds':reference, 'stages':times, 'gpus':[6]}}
for name in ('fast-v1','fast-v2','fast-overlap','fast-dual','tensorcore','fast-v3'):
    data = json.loads((p/name/'timing.json').read_text())
    data['speedup_vs_reference'] = reference / data['wall_seconds']
    output[name]=data
(p/'results_48.json').write_text(json.dumps(output,indent=2)+'\n')
print('baseline',reference)
for name, data in output.items():
    if name!='baseline': print(name, round(data['wall_seconds'],2), round(data['speedup_vs_reference'],2))
