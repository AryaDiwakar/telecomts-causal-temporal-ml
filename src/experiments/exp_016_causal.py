"""exp_016 - Phase 3: causal intervention fidelity (smoke)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np, torch
from src.models.causal_intervention import interventional_fidelity
from src.utils.registry import append
INTERIM=ROOT/"data"/"interim"; PROCESSED=ROOT/"data"/"processed"; RES=ROOT/"outputs"/"results"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args()
    # smoke: random anomaly fn, report fidelity on a sample
    rng=np.random.default_rng(args.seed)
    X=np.load(INTERIM/"chunks_X.npy").astype("float32")[:64]
    def afn(x):
        return torch.tensor(x.reshape(x.shape[0],-1)[:, :4].mean(axis=1), dtype=torch.float32)
    xt=torch.tensor(X)
    fid=interventional_fidelity(lambda t: afn(t.numpy()), xt, [0,1], [2,3])
    print(f"interventional fidelity (smoke) = {fid:.4f}")
    out={"experiment":"exp_016","phase":3,"seed":args.seed,"metrics":{"interventional_fidelity":fid}}
    RES.mkdir(parents=True,exist_ok=True); (RES/f"exp_016_causal_seed{args.seed}.json").write_text(json.dumps(out,indent=2))
    try: append({"experiment":"exp_016","seed":args.seed,"status":"COMPLETE","metrics":out["metrics"],"task":"phase3_causal"})
    except Exception as e: print("registry skip",e)
if __name__=="__main__": main()
