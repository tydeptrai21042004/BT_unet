#!/usr/bin/env python3
"""CUDA-synchronized inference latency and memory from authentic checkpoints."""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from common import load_checkpoint,write_json
from src.engine.output_utils import parse_model_output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--config')
    p.add_argument('--model')
    p.add_argument('--image-size',type=int,default=352)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--repetitions',type=int,default=100)
    p.add_argument('--device',default='auto')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    model,cfg,name=load_checkpoint(args.checkpoint,args.model,args.config,device)
    in_channels=int(cfg.get('model',{}).get('in_channels',3))
    tensor=torch.randn(args.batch_size,in_channels,args.image_size,args.image_size,device=device)
    sync=lambda:torch.cuda.synchronize(device) if str(device).startswith('cuda') else None
    with torch.inference_mode():
        for _ in range(args.warmup):parse_model_output(model(tensor))
        sync()
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
        values=[]
        for _ in range(args.repetitions):
            sync();begin=time.perf_counter()
            parse_model_output(model(tensor))
            sync();values.append((time.perf_counter()-begin)*1000)
    vals=sorted(values)
    report={'model':name,'device':str(device),'image_size':args.image_size,
            'batch_size':args.batch_size,'warmup':args.warmup,'repetitions':args.repetitions,
            'latency_ms_median_batch':statistics.median(vals),
            'latency_ms_p95_batch':vals[min(len(vals)-1,int(.95*len(vals)))],
            'fps_images_per_second':1000*args.batch_size/statistics.mean(vals),
            'parameters_total':sum(p.numel() for p in model.parameters()),
            'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'gpu_peak_allocated_mib':(torch.cuda.max_memory_allocated(device)/1048576 if str(device).startswith('cuda') else None)}
    write_json(report,args.output)
    print(report)
