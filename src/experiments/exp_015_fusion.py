"""exp_015 - Phase 2: multimodal fusion detection (smoke)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from src.models.fusion import MultimodalFusion
from src.utils.registry import append
INTERIM=ROOT/"data"/"interim"; PROCESSED=ROOT/"data"/"processed"; RES=ROOT/"outputs"/"results"
def load():
    X=np.load(INTERIM/"chunks_X.npy"); scaler=json.loads((PROCESSED/"scaler.json").read_text())
    loc=np.asarray(scaler["loc"])[:16]; scale=np.asarray(scaler["scale"])[:16]
    Xs=X.copy().astype("float32"); Xs[:,:,:16]=(Xs[:,:,:16]-loc)/scale
    idx=np.load(PROCESSED/"split_indices.npz"); tar=dict(np.load(PROCESSED/"targets.npz"))
    return Xs,idx,tar
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--epochs",type=int,default=1); ap.add_argument("--batch",type=int,default=64); args=ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    Xs,idx,tar=load(); tr,te=idx["train"],idx["test"]
    Xtr=torch.tensor(Xs[tr]); Xte=torch.tensor(Xs[te]); ytr=tar["y_anomaly"][tr].astype("float32"); yte=tar["y_anomaly"][te].astype("float32")
    n=Xs.shape[2]
    m=MultimodalFusion(n_channels=n,hidden=32,n_heads=2,n_context=4,n_graph_nodes=4)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); lf=nn.BCEWithLogitsLoss(); head=nn.Linear(32,1)
    params=list(m.parameters())+list(head.parameters()); opt=torch.optim.Adam(params,lr=1e-3)
    gtr=torch.zeros(len(tr),4,4); gte=torch.zeros(len(te),4,4); ctr=torch.zeros(len(tr),4); cte=torch.zeros(len(te),4)
    for ep in range(args.epochs):
        m.train(); head.train()
        perm=torch.randperm(len(Xtr))
        for i in range(0,len(Xtr),args.batch):
            sl=perm[i:i+args.batch]
            opt.zero_grad()
            f=m(Xtr[sl],gtr[sl],ctr[sl]); logits=head(f).squeeze(-1)
            loss=lf(logits,torch.tensor(ytr[sl]))
            loss.backward(); opt.step()
    m.eval(); head.eval()
    with torch.no_grad():
        f=m(Xte,gte,cte); prob=torch.sigmoid(head(f).squeeze(-1)).numpy()
    pr=float(average_precision_score(yte,prob)); roc=float(roc_auc_score(yte,prob))
    print(f"fusion TEST PR={pr:.4f} ROC={roc:.4f}")
    out={"experiment":"exp_015","phase":2,"seed":args.seed,"metrics":{"pr_auc":pr,"roc_auc":roc}}
    RES.mkdir(parents=True,exist_ok=True); (RES/f"exp_015_fusion_seed{args.seed}.json").write_text(__import__("json").dumps(out,indent=2))
    try: append({"experiment":"exp_015","seed":args.seed,"status":"COMPLETE","metrics":out["metrics"],"task":"phase2_fusion"})
    except Exception as e: print("registry skip",e)
if __name__=="__main__": main()
