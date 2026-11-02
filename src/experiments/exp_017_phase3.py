"""exp_017 - Phase 3: causal intervention model (smoke + train).

Trains a backbone with anomaly + RCA heads and a causal-consistency loss,
then reports interventional RCA fidelity.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from src.models.backbone import TemporalBackbone
from src.models.causal_intervention import CausalConsistencyLoss, interventional_fidelity, intervene
from src.utils.registry import append
INTERIM=ROOT/"data"/"interim"; PROCESSED=ROOT/"data"/"processed"; RES=ROOT/"outputs"/"results"

def load():
    X=np.load(INTERIM/"chunks_X.npy"); scaler=json.loads((PROCESSED/"scaler.json").read_text())
    loc=np.asarray(scaler["loc"])[:16]; scale=np.asarray(scaler["scale"])[:16]
    Xs=X.copy().astype("float32"); Xs[:,:,:16]=(Xs[:,:,:16]-loc)/scale
    idx=np.load(PROCESSED/"split_indices.npz"); tar=dict(np.load(PROCESSED/"targets.npz"))
    return Xs,idx,tar

class Phase3Model(nn.Module):
    def __init__(self, n_channels, hidden=64):
        super().__init__()
        self.bb=TemporalBackbone(n_channels=n_channels,hidden=hidden,n_heads=4,n_tcn_layers=3,n_transformer_layers=2,dropout=0.2)
        self.ano=nn.Linear(hidden,1)
        self.rca=nn.Linear(hidden,16)
    def forward(self,x):
        h=self.bb(x)
        return self.ano(h).squeeze(-1), self.rca(h), h

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--batch",type=int,default=64); args=ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    Xs,idx,tar=load(); tr,te=idx["train"],idx["test"]
    Xtr=torch.tensor(Xs[tr]); Xte=torch.tensor(Xs[te])
    ytr=tar["y_anomaly"][tr].astype("float32"); yte=tar["y_anomaly"][te].astype("float32")
    afftr=tar["y_affected"][tr].astype("float32")
    model=Phase3Model(Xs.shape[2]); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    lf=nn.BCEWithLogitsLoss(); rl=nn.BCEWithLogitsLoss(); ccl=CausalConsistencyLoss(2.0)
    for ep in range(args.epochs):
        model.train(); tot=0.0
        perm=torch.randperm(len(Xtr))
        for i in range(0,len(Xtr),args.batch):
            sl=perm[i:i+args.batch]
            xb=Xtr[sl]; yb=torch.tensor(ytr[sl]); ab=torch.tensor(afftr[sl])
            opt.zero_grad()
            al,rlg,_=model(xb)
            loss=lf(al,yb)+rl(rlg,ab)+ccl(rlg,xb,None,None)
            loss.backward(); opt.step(); tot+=float(loss.item())*len(sl)
        print(f"epoch {ep+1}/{args.epochs} loss={tot/len(Xtr):.4f}")
    model.eval()
    with torch.no_grad():
        al,rlg,_=model(Xte)
        prob=torch.sigmoid(al).numpy()
    pr=float(average_precision_score(yte,prob)); roc=float(roc_auc_score(yte,prob))
    # Interventional fidelity (pilot): use the anomaly score and the top-k RCA
    # head as the candidate root set.  The control must be a *matched random
    # non-root* set of the same size; the earlier pilot accidentally reused the
    # same ordered top-k set for both sides, which makes the score meaningless.
    # The validated, multi-seed evaluation with positive/negative controls lives
    # in scripts/validate_fidelity.py.
    def afn(xt):
        with torch.no_grad():
            a,_ ,_=model(torch.tensor(xt,dtype=torch.float32)); return torch.sigmoid(a)
    N=min(200,len(Xte))
    rng=np.random.default_rng(args.seed)
    roots=[list(np.argsort(-rlg[i].numpy())[:3]) for i in range(N)]
    nonr=[]
    for r in roots:
        pool=[c for c in range(16) if c not in r]
        nonr.append(rng.choice(pool,size=len(r),replace=False).tolist())
    Xsmp=Xte[:N].numpy()
    fids=[interventional_fidelity(lambda t: afn(t).numpy(), torch.tensor(Xsmp[i:i+1]), roots[i], nonr[i]) for i in range(N)]
    fid=float(np.mean(fids))
    print(f"TEST PR={pr:.4f} ROC={roc:.4f} fidelity(pilot)={fid:.4f}")
    out={"experiment":"exp_017","phase":3,"seed":args.seed,"metrics":{"pr_auc":pr,"roc_auc":roc,"interventional_fidelity":fid}}
    RES.mkdir(parents=True,exist_ok=True); (RES/f"exp_017_phase3_seed{args.seed}.json").write_text(json.dumps(out,indent=2))
    try: append({"experiment":"exp_017","seed":args.seed,"status":"COMPLETE","metrics":out["metrics"],"task":"phase3_causal"})
    except Exception as e: print("registry skip",e)
if __name__=="__main__": main()
