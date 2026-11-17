#!/usr/bin/env python3
"""Reviewer-ready input/GT/baseline/BT masks and boundary zoom-in panels.

Prediction files MUST exist and share the same sample ID; this script never
creates or selects favorable predictions. Cases can be predefined by ID or a
selection JSON produced by select_qualitative_cases.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .common import write_json, sha256_file
except ImportError:
    from common import write_json, sha256_file
from src.datasets import build_dataset


def boundary_crop(gt: np.ndarray, width: int = 112) -> tuple[int,int,int,int]:
    """Deterministic boundary-centered ROI, independent of model predictions."""
    from scipy.ndimage import binary_erosion
    h,w=gt.shape
    if gt.any():
        edge = gt & ~binary_erosion(gt)
        coords = np.argwhere(edge)
        # Near median boundary point avoids selecting BT-favorable cases.
        cy,cx = coords[len(coords)//2] if len(coords) else np.argwhere(gt).mean(axis=0)
    else:
        cy,cx = (h-1)/2,(w-1)/2
    side=min(max(8,int(width)),h,w)
    y0=max(0,min(h-side,int(round(cy-side/2))))
    x0=max(0,min(w-side,int(round(cx-side/2))))
    return (x0,y0,x0+side,y0+side)


def _overlay(img: np.ndarray,gt: np.ndarray,pred: np.ndarray,
             *, overlay_errors=True) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    view=img.astype(np.float32).copy()
    if overlay_errors:
        fp=pred & ~gt;fn=gt & ~pred
        view[fp]=.34*view[fp]+.66*np.array([220,45,40])
        view[fn]=.34*view[fn]+.66*np.array([25,135,220])
    else:
        view[pred]=.65*view[pred]+.35*np.array([35,170,80])
    edge=pred & ~binary_erosion(pred)
    view[edge]=np.array([45,225,75])
    return np.clip(view,0,255).astype(np.uint8)


def _read_case_labels(path):
    if not path:return {},None
    p=Path(path); payload=json.loads(p.read_text(encoding='utf-8'))
    cases=payload.get('cases',[])
    return {str(row['sample_id']):row.get('selection_reason','') for row in cases},p


def make_panel(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    data=build_dataset(args.dataset,args.data_root,split=args.split,
                       image_size=args.image_size,return_paths=True)
    data_map={str(sid):(image,mask) for sid,image,mask in data.samples}
    models=[x.strip() for x in args.model_names.split(',') if x.strip()]
    dirs=[Path(x.strip()) for x in args.pred_dirs.split(',') if x.strip()]
    if not models or len(models)!=len(dirs) or len(set(models))!=len(models):
        raise ValueError('Require unique model names and exactly one folder per model')
    tags,case_json=_read_case_labels(args.cases_json)
    if args.ids and args.cases_json:
        raise ValueError('Use --ids OR --cases-json for deterministic case selection')
    ids=([x.strip() for x in args.ids.split(',') if x.strip()] if args.ids else
         list(tags) if args.cases_json else sorted(data_map)[:args.max_cases])
    if len(ids)>args.max_cases:ids=ids[:args.max_cases]
    if not ids or len(ids)!=len(set(ids)) or any(s not in data_map for s in ids):
        raise ValueError('IDs must exist uniquely in requested dataset split')
    ncol=2+len(models); nrow=len(ids)*(2 if args.zoom else 1)
    fig,axes=plt.subplots(nrow,ncol,figsize=(2.65*ncol,2.6*nrow),squeeze=False)
    details=[]
    for i,sid in enumerate(ids):
        img_path,mask_path=data_map[sid]
        with Image.open(img_path) as im: rgb=np.asarray(im.convert('RGB'))
        h,w=rgb.shape[:2]
        with Image.open(mask_path) as im:
            gt=np.asarray(im.convert('L').resize((w,h),Image.Resampling.NEAREST))>=128
        roi=boundary_crop(gt,args.zoom_pixels)
        panels=[rgb,np.stack([gt.astype(np.uint8)*255]*3,-1)]
        sources={}
        for model,folder in zip(models,dirs):
            pred_file=folder/f'{sid}.png'
            if not pred_file.is_file():
                raise FileNotFoundError(f'Missing {model} prediction: {pred_file}')
            with Image.open(pred_file) as im:
                # 16-bit probability exports are not accepted as hard masks;
                # report requires the explicit binary masks output directory.
                if im.mode in {'I;16','I;16B','I'}:
                    raise ValueError(f'Expected binary 8-bit mask, got {im.mode}: {pred_file}')
                pred=np.asarray(im.convert('L').resize((w,h),Image.Resampling.NEAREST))>=128
            panels.append(_overlay(rgb,gt,pred,overlay_errors=not args.no_error_overlay))
            sources[model]={'prediction_path':str(pred_file),
                            'prediction_sha256':sha256_file(pred_file)}
        main=i*(2 if args.zoom else 1)
        for col,arr in enumerate(panels):
            ax=axes[main,col];ax.imshow(arr);ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title((['Input','Ground truth']+models)[col],fontsize=11)
            if args.zoom:
                x0,y0,x1,y1=roi
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,edgecolor='white',
                                       facecolor='none',linewidth=1.3))
                sub=axes[main+1,col];sub.imshow(arr[y0:y1,x0:x1]);sub.set_xticks([]);sub.set_yticks([])
                if col==0:sub.set_ylabel('Boundary zoom',fontsize=9)
        axes[main,0].set_ylabel(f'{sid}\n{tags.get(sid,"")}',fontsize=9)
        details.append({'sample_id':sid,'selection_reason':tags.get(sid,'user-specified' if args.ids else 'alphabetical'),
                        'image':str(img_path),'image_sha256':sha256_file(img_path),
                        'ground_truth':str(mask_path),'mask_sha256':sha256_file(mask_path),
                        'roi_xyxy':roi if args.zoom else None,'prediction_sources':sources})
    fig.tight_layout(pad=.65)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=args.dpi,bbox_inches='tight',facecolor='white');plt.close(fig)
    write_json({'dataset':args.dataset,'split':args.split,'cases_json':str(case_json) if case_json else None,
                'items':details,'models':models,'dpi':args.dpi,'zoom':bool(args.zoom),
                'legend':{'false_positive':'red','false_negative':'blue','predicted_boundary':'green'}},
               out.with_suffix('.json'))
    return out


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--data-root',default='data')
    p.add_argument('--split',default='test');p.add_argument('--image-size',type=int,default=352)
    p.add_argument('--model-names',required=True)
    p.add_argument('--pred-dirs',required=True,help='Comma-separated folders of 8-bit <sample_id>.png hard masks')
    p.add_argument('--ids');p.add_argument('--cases-json')
    p.add_argument('--max-cases',type=int,default=5)
    p.add_argument('--zoom',action='store_true',help='Second row per case, predetermined GT-boundary crop')
    p.add_argument('--zoom-pixels',type=int,default=112)
    p.add_argument('--no-error-overlay',action='store_true')
    p.add_argument('--dpi',type=int,default=350)
    p.add_argument('--output',required=True)
    return p

if __name__=='__main__':
    a=parser().parse_args()
    if a.dpi<300:raise ValueError('Reviewer 7 requires at least 300 dpi')
    print('Saved',make_panel(a))
