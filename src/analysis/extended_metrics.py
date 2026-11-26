"""Prediction-based metrics; no extra model inference or training needed."""
import numpy as np

def confusion_metrics(tp, fp, tn, fn):
    denom = float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn)
    return {'mcc':(tp*tn-fp*fn)/np.sqrt(denom) if denom else float('nan'),
            'foreground_area_error':abs(fp-fn)/max(tp+fp+tn+fn,1),
            'tp':tp,'fp':fp,'tn':tn,'fn':fn}

def _ece(p,y,bins=15):
    if not p.size: return float('nan')
    ids=np.minimum((p*bins).astype(int),bins-1)
    counts=np.bincount(ids,minlength=bins)
    return float(np.abs(np.bincount(ids,weights=p,minlength=bins)-np.bincount(ids,weights=y,minlength=bins)).sum()/counts.sum())

def probability_metrics(prob,truth):
    p=np.asarray(prob,dtype=np.float64).ravel(); y=np.asarray(truth,dtype=bool).ravel()
    q=np.clip(p,1e-7,1-1e-7)
    pos=p>=.5
    # Noninterpolated AP: precision at distinct thresholds, weighted by recall increments.
    ap=float('nan')
    if y.any():
        order=np.argsort(-p,kind='stable'); ps=p[order]; ys=y[order]
        ends=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]), len(ps)-1]
        tps=np.cumsum(ys)[ends]; ap=float(np.sum(np.diff(np.r_[0,tps])*(tps/(ends+1)))/y.sum())
    return {'brier':float(np.mean((p-y)**2)), 'nll':float(-np.mean(np.where(y,np.log(q),np.log1p(-q)))),
            'ece':_ece(p,y), 'foreground_prediction_ece':_ece(p[pos],y[pos]), 'auprc':ap}
