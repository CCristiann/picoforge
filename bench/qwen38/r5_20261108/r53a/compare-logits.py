"""Controller helper: G3-style logits report of ACTUAL against REFERENCE ([n][vocab] f32 files)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from qwen38_g3 import logits_report, GATE
a, r = (np.fromfile(p, '<f4') for p in sys.argv[1:3]); vocab = int(sys.argv[3])
rep = logits_report(a.reshape(-1, vocab), r.reshape(-1, vocab).astype(np.float64))
short = {k: rep[k] for k in ('scaled', 'absolute', 'argmax_matches', 'margin_decided', 'top5_order_matches')}
short['rows'] = len(rep['per_token_scaled']); short['gate'] = GATE; short['pass'] = rep['scaled'] <= GATE and rep['argmax_matches'] == short['rows']
print(json.dumps(short))
if len(sys.argv) > 4: Path(sys.argv[4]).write_text(json.dumps(rep) + '\n')
