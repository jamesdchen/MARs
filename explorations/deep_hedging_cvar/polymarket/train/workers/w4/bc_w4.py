"""BC warm start from the w4 learned quoter: train/bc.py unchanged, with its teacher replaced.

    python bc_w4.py --quoter runs/q1_best.json --model-idx 0 -- --num-envs 1024 --steps 2048 \
        --value --hidden 64 --out ckpt/bc_m0.pt [other train/bc.py args]

The teacher is quoter.Quoter(models[model_idx], **params) on the collection env's observations
(train split only, as bc.py). Everything else (collection, fit, checkpoint format) is bc.py's.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, TRAIN)
sys.path.insert(0, HERE)

argv = sys.argv[1:]
split = argv.index('--') if '--' in argv else len(argv)
own, rest = argv[:split], argv[split + 1:]
qpath = own[own.index('--quoter') + 1]
midx = int(own[own.index('--model-idx') + 1]) if '--model-idx' in own else 0

import bc  # noqa: E402
import envs  # noqa: E402
from quoter import Quoter  # noqa: E402

spec = json.load(open(qpath))
model = spec['models'][midx]
if not os.path.isabs(model):
    model = os.path.join(HERE, model)
params = spec['params']
quoter = Quoter(model, **params)


def teacher(self, name=None, params_=None, env_kw=None):
    self.teacher_params = {'w4_quoter': params, 'model': os.path.relpath(model, HERE), 'quoter_json': qpath}
    return quoter


envs.RealAdapter.teacher = teacher
sys.argv = ['bc.py', '--teacher', 'w4_quoter', '--teacher-source', 'tuned', '--env', 'real'] + rest
bc.main()
