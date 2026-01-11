import torch
import pprint
from pathlib import Path

pth = Path('model/plant_disease.pth')
print('Inspecting', pth)
if not pth.exists():
    print('File not found:', pth)
    raise SystemExit(1)

data = torch.load(str(pth), map_location='cpu')
print('Top-level type:', type(data))

if isinstance(data, dict):
    keys = list(data.keys())
    print('Top-level keys (count={}):'.format(len(keys)))
    pprint.pprint(keys[:200])
    # show types of some keys
    for k in keys[:200]:
        v = data[k]
        t = type(v)
        shape = getattr(v, 'shape', None)
        print(f"- {k}: {t}, shape={shape}")
    # check common nested candidates
    for candidate in ('state_dict', 'model_state_dict', 'model', 'weights'):
        if candidate in data:
            print('\nFound candidate key:', candidate, 'type=', type(data[candidate]))
            if isinstance(data[candidate], dict):
                print('  nested keys sample:')
                pprint.pprint(list(data[candidate].keys())[:200])
else:
    print('Object repr:')
    pprint.pprint(data)
