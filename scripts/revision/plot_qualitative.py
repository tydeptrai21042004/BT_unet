#!/usr/bin/env python3
"""Create aligned image/GT/baseline/BT panels from exported masks.

All model masks must have matching sample IDs. No generated prediction is invented.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from common import write_json
from src.datasets import build_dataset


def make_panel(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = build_dataset(args.dataset, args.data_root, split=args.split, image_size=args.image_size,
                         return_paths=True)
    data_map = {sid: (image,mask) for sid,image,mask in data.samples}
    models = [x.strip() for x in args.model_names.split(',')]
    dirs = [Path(x.strip()) for x in args.pred_dirs.split(',')]
    if len(models) != len(dirs):
        raise ValueError('model-names and pred-dirs must have equal length')
    ids = [x.strip() for x in args.ids.split(',')] if args.ids else list(data_map)[:args.max_cases]
    if not ids or any(s not in data_map for s in ids):
        raise ValueError('Requested sample IDs missing from dataset split')
    figure, axes = plt.subplots(len(ids), 2 + len(models), figsize=(2.7*(2+len(models)), 2.8*len(ids)),squeeze=False)
    details=[]
    for row,sid in enumerate(ids):
        image_path,gt_path=data_map[sid]
        img=Image.open(image_path).convert('RGB')
        gt=np.asarray(Image.open(gt_path).convert('L').resize(img.size, Image.Resampling.NEAREST)) >= 128
        axes[row,0].imshow(img);axes[row,0].set_ylabel(sid, fontsize=8)
        axes[row,1].imshow(gt,cmap='gray',vmin=0,vmax=1)
        details.append({'sample_id':sid,'image':str(image_path),'ground_truth':str(gt_path)})
        for col,(model,maskdir) in enumerate(zip(models,dirs),start=2):
            path=maskdir/f'{sid}.png'
            if not path.exists():
                raise FileNotFoundError(f'Missing prediction: {path}')
            with Image.open(path) as pred_image:
                pred=np.asarray(pred_image.convert('L').resize(img.size,Image.Resampling.NEAREST)) >= 128
            axes[row,col].imshow(img,alpha=.55)
            overlay=np.zeros((*pred.shape,4),dtype=float)
            overlay[pred & ~gt]=[1,0,0,.72]   # false positive red
            overlay[~pred & gt]=[0,.75,1,.72]  # false negative blue
            axes[row,col].imshow(overlay)
            axes[row,col].contour(pred,levels=[.5],colors=['lime'],linewidths=.65)
        for col,ax in enumerate(axes[row]):
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title((['Input','Ground truth']+models)[col])
    figure.tight_layout()
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(out,dpi=220,bbox_inches='tight');plt.close(figure)
    write_json({'dataset':args.dataset,'split':args.split,'items':details,'models':models,
                'colors':{'FP':'red','FN':'cyan','predicted_boundary':'lime'}},out.with_suffix('.json'))
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--data-root',default='data')
    p.add_argument('--split',default='test')
    p.add_argument('--image-size',type=int,default=352)
    p.add_argument('--model-names',required=True,help='Comma-separated labels')
    p.add_argument('--pred-dirs',required=True,help='Comma-separated directories containing <ID>.png')
    p.add_argument('--ids',help='Comma-separated preselected sample IDs')
    p.add_argument('--max-cases',type=int,default=5)
    p.add_argument('--output',required=True)
    make_panel(p.parse_args())
