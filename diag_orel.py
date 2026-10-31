import json, os, sys, collections
import numpy as np
sys.path.insert(0, "."); import orel
import os
D=os.environ.get("ARC_DATA","../ARC-AGI-2/data")+"/training/"
ids=json.load(open("dev700.json"))[:300]
reasons=collections.Counter(); unexpl=[]
for tid in ids:
    t=json.load(open(D+tid+".json"))
    pairs=[(np.array(p["input"]),np.array(p["output"])) for p in t["train"]]
    if not all(a.shape==b.shape for a,b in pairs): reasons["diff_shape"]+=1; continue
    best="no_ctx"
    for mode in orel.MODES:
        ctxs=[orel.Ctx(a,mode) for a,b in pairs]
        if not all(c.ok for c in ctxs): r="too_many_or_no_objects"
        else:
            rel=[{k:orel.rel_colour(c,k) for k in orel.RELS} for c in ctxs]
            A=np.concatenate([orel.consistent_actions(c,o,rc) for c,(a,o),rc in zip(ctxs,pairs,rel)],0)
            if not (~A[:,0]).any(): r="no_object_change"
            elif not A.any(1).all():
                r="unexplainable_object"; frac=(~A.any(1)).mean()
                unexpl.append(frac)
            else:
                r="explainable"
        # priority: explainable > unexplainable > others
        order=["explainable","unexplainable_object","no_object_change","too_many_or_no_objects","no_ctx"]
        if order.index(r)<order.index(best): best=r
    reasons[best]+=1
print(reasons); print("mean frac unexplained objects (when unexplainable):",np.mean(unexpl) if unexpl else None)
