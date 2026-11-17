#!/usr/bin/env python3
"""Genuine 3D CT/MRI binary segmentation: training, sliding-window evaluation.

No precomputed or synthetic benchmark results. For controlled comparison, train
--model unet3d and --model bt_unet3d with identical patient-disjoint manifests,
seed, optimizer, patch shape and epochs. The code never auto-downloads datasets.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.datasets.medical_volume_dataset import MedicalVolumeDataset,audit_patient_splits
from src.models.proposal.bt_unet3d import UNet3D,BTUNet3D
from scripts.revision.common import sha256_file,write_json,write_csv


def build_3d(name,channels=(16,32,64,128)):
    if name=='unet3d':return UNet3D(channels=channels)
    if name=='bt_unet3d':return BTUNet3D(channels=channels)
    raise ValueError('Model must be unet3d or bt_unet3d')


def sliding_starts(n:int,size:int,overlap:float):
    if not 0<=overlap<1:raise ValueError('overlap must be in [0,1)')
    if n<=size:return [0]
    stride=max(1,round(size*(1-overlap)))
    starts=list(range(0,n-size+1,stride))
    if starts[-1]!=n-size:starts.append(n-size)
    return starts


@torch.inference_mode()
def predict_volume(model,image,patch=(64,64,64),overlap=.25):
    """Volume-wise probability average of overlapping 3D logits patches."""
    if image.ndim!=5 or image.shape[0]!=1:
        raise ValueError('Expect one whole volume [1,C,D,H,W]')
    patch=tuple(int(v) for v in patch)
    shape=image.shape[-3:]
    pads=tuple(max(0,p-s) for p,s in zip(patch,shape))
    x=F.pad(image,(0,pads[2],0,pads[1],0,pads[0]))
    d,h,w=x.shape[-3:]
    acc=torch.zeros((1,1,d,h,w),dtype=torch.float32,device=image.device)
    count=torch.zeros_like(acc)
    for zz in sliding_starts(d,patch[0],overlap):
        for yy in sliding_starts(h,patch[1],overlap):
            for xx in sliding_starts(w,patch[2],overlap):
                window=x[...,zz:zz+patch[0],yy:yy+patch[1],xx:xx+patch[2]]
                probs=model(window).float().sigmoid()
                acc[...,zz:zz+patch[0],yy:yy+patch[1],xx:xx+patch[2]]+=probs
                count[...,zz:zz+patch[0],yy:yy+patch[1],xx:xx+patch[2]]+=1
    if not torch.all(count>0):raise RuntimeError('Sliding window failed to cover volume')
    return (acc/count)[...,:shape[0],:shape[1],:shape[2]]


def segmentation_metrics(prob,mask,spacing,threshold=.5):
    """3D voxel-level binary Dice/IoU and physical HD95 in millimeters."""
    from scipy import ndimage
    pred=np.asarray(prob)>=threshold
    gt=np.asarray(mask)>=.5
    if pred.shape!=gt.shape or pred.ndim!=3:raise ValueError('Volume mask mismatch')
    tp=int(np.sum(pred&gt));fp=int(np.sum(pred&~gt));fn=int(np.sum(~pred&gt))
    dice=(2*tp+1)/(2*tp+fp+fn+1)
    iou=(tp+1)/(tp+fp+fn+1)
    mae=float(np.abs(np.asarray(prob)-gt.astype(np.float32)).mean())
    spacing=tuple(float(x) for x in spacing)
    if not pred.any() and not gt.any(): hd95=0.
    elif not pred.any() or not gt.any():hd95=float('nan')
    else:
        structure=ndimage.generate_binary_structure(3,1)
        bpred=pred & ~ndimage.binary_erosion(pred,structure=structure)
        bgt=gt & ~ndimage.binary_erosion(gt,structure=structure)
        to_gt=ndimage.distance_transform_edt(~bgt,sampling=spacing)[bpred]
        to_pred=ndimage.distance_transform_edt(~bpred,sampling=spacing)[bgt]
        hd95=float(np.percentile(np.concatenate([to_gt,to_pred]),95))
    return {'dice':float(dice),'iou':float(iou),
            'precision':float(tp/(tp+fp+1e-7)),'recall':float(tp/(tp+fn+1e-7)),
            'mae':mae,'hd95_mm':hd95,'volume_voxels':int(gt.size),
            'foreground_voxels':int(gt.sum())}


def evaluate(model,data,device,patch,overlap,threshold=.5):
    model.eval();results=[]
    with torch.inference_mode():
        for sample in data:
            image=sample['image'][None].to(device)
            prob=predict_volume(model,image,patch,overlap)[0,0].cpu().numpy()
            mask=sample['mask'][0].numpy()
            metrics=segmentation_metrics(prob,mask,sample['spacing'].numpy(),threshold)
            if not sample['spacing_is_physical']:
                metrics['hd95_voxel_units']=metrics.pop('hd95_mm')
                metrics['hd95_mm']=float('nan')
            else:
                metrics['hd95_voxel_units']=float('nan')
            results.append({'sample_id':sample['id'],'patient_id':sample['patient_id'],
                            'physical_spacing_known':bool(sample['spacing_is_physical']),**metrics})
    if not results:raise ValueError('Empty 3D evaluation')
    summary={'n_volumes':len(results),'n_patients':len({r['patient_id'] for r in results}),
             'metric_means':{metric:float(np.nanmean([r[metric] for r in results])) if
                             np.isfinite([r[metric] for r in results]).any() else None
                             for metric in ('dice','iou','precision','recall','mae','hd95_mm','hd95_voxel_units')}}
    return results,summary


def bce_dice(logits,mask):
    bce=F.binary_cross_entropy_with_logits(logits,mask)
    probs=torch.sigmoid(logits)
    axes=tuple(range(2,probs.ndim))
    dice=(2*(probs*mask).sum(dim=axes)+1)/((probs+mask).sum(dim=axes)+1)
    return .5*bce+.5*(1-dice.mean())


def run(args):
    if any(x<2**(len(args.channels)-1) for x in args.patch):
        raise ValueError('Patch must fit U-Net downsampling depth')
    train_file=Path(args.manifest_dir)/'train.csv'
    val_file=Path(args.manifest_dir)/'val.csv'
    test_file=Path(args.manifest_dir)/'test.csv'
    audit=audit_patient_splits(train_file,val_file,test_file)
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    provenance={'model':args.model,'seed':args.seed,'device':args.device,
                'optimizer':'AdamW','learning_rate':args.lr,'weight_decay':args.weight_decay,
                'loss':'0.5*BCEWithLogits+0.5*softDice', 'epochs':args.epochs,
                'patch':args.patch,'overlap':args.overlap,'channels':args.channels,
                'splits':audit,'manifest_sha256':{split:sha256_file(path) for split,path
                    in [('train',train_file),('val',val_file),('test',test_file)]},
                'metric_units':{'hd95_mm':'mm only for NIfTI or NPZ with verified spacing; see metadata warning',
                                'hd95_voxel_units':'voxel units for NPY without physical spacing'},
                'note':'Volumetric 3D extension, not submitted 2D model; train on patient-disjoint source volumes.'}
    manifest_out=output/'experiment_manifest.json'
    stable_keys=('model','seed','patch','channels','manifest_sha256')
    if manifest_out.is_file():
        previous=json.loads(manifest_out.read_text(encoding='utf-8'))
        if any(previous.get(key)!=provenance.get(key) for key in stable_keys):
            raise ValueError('Experiment provenance conflict; select a fresh output-dir')
    elif args.eval_only:
        raise FileNotFoundError('Evaluation requires the original training experiment_manifest.json')
    if args.eval_only:
        write_json({'evaluation_device_requested':args.device,'training_provenance':str(manifest_out),
                    'checkpoint':str(output/'checkpoints'/'best.pt')},
                   output/'evaluation_manifest.json')
    else:
        if manifest_out.is_file() and json.loads(manifest_out.read_text(encoding='utf-8'))!=provenance:
            raise ValueError('Training settings changed; use a fresh output-dir')
        write_json(provenance,manifest_out)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    model=build_3d(args.model,tuple(args.channels)).to(device)
    train=MedicalVolumeDataset(args.data_root,train_file,patch_shape=args.patch,train=True,seed=args.seed)
    val=MedicalVolumeDataset(args.data_root,val_file)
    test=MedicalVolumeDataset(args.data_root,test_file)
    best_path=output/'checkpoints'/'best.pt'
    if args.eval_only:
        if not best_path.is_file():raise FileNotFoundError(best_path)
        ckpt=torch.load(best_path,map_location=device,weights_only=False)
        recorded=ckpt.get('provenance') or {}
        if any(recorded.get(key)!=provenance.get(key) for key in stable_keys):
            raise ValueError('Checkpoint architecture, dataset or seed provenance mismatch')
        model.load_state_dict(ckpt['state_dict'])
    else:
        if args.epochs<1:raise ValueError('Training epochs must be positive')
        train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,
                                num_workers=0,generator=torch.Generator().manual_seed(args.seed))
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*.01)
        history=[];best=-float('inf')
        for epoch in range(1,args.epochs+1):
            model.train();losses=[]
            for batch in train_loader:
                img=batch['image'].to(device);mask=batch['mask'].to(device)
                optimizer.zero_grad(set_to_none=True)
                pred=model(img);loss=bce_dice(pred,mask)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite 3D training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step();losses.append(float(loss.detach().cpu()))
            schedule.step()
            _,metrics=evaluate(model,val,device,args.patch,args.overlap)
            dice=metrics['metric_means']['dice']
            history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val_dice':dice,
                            'learning_rate':optimizer.param_groups[0]['lr']})
            if dice>best:
                best=dice;best_path.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'state_dict':model.state_dict(),'epoch':epoch,'val_dice':dice,
                            'provenance':provenance},best_path)
            print(f'[{args.model} {epoch}/{args.epochs}] loss={history[-1]["train_loss"]:.4f} val_dice={dice:.4f}',flush=True)
        write_csv(history,output/'history.csv')
        model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)['state_dict'])
    val_rows,val_summary=evaluate(model,val,device,args.patch,args.overlap)
    test_rows,test_summary=evaluate(model,test,device,args.patch,args.overlap)
    for name,rows,summary in [('val',val_rows,val_summary),('test',test_rows,test_summary)]:
        write_csv(rows,output/f'{name}_per_volume.csv')
        summary.update({'checkpoint_sha256':sha256_file(best_path),'model':args.model,'seed':args.seed})
        write_json(summary,output/f'{name}_summary.json')
    return test_summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['unet3d','bt_unet3d'],required=True)
    p.add_argument('--data-root',required=True);p.add_argument('--manifest-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--patch',type=int,nargs=3,default=(64,64,64),metavar=('D','H','W'))
    p.add_argument('--channels',type=int,nargs='+',default=(16,32,64,128))
    p.add_argument('--overlap',type=float,default=.25)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--weight-decay',type=float,default=.0001)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='auto')
    p.add_argument('--eval-only',action='store_true')
    return p

if __name__=='__main__':
    a=parser().parse_args();print(json.dumps(run(a),indent=2))
