#!/usr/bin/env python3
"""Checkpoint-grounded, synchronized inference benchmarking.

Reports measured parameters, median/p95 latency, throughput and CUDA memory.
Convolution/linear MACs are counted by forward hooks. FFT butterfly complexity
is reported SEPARATELY, never silently omitted or disguised as exact FLOPs.
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import torch
from torch import nn

try:
    from .common import load_checkpoint, write_json, sha256_file
except ImportError:
    from common import load_checkpoint, write_json, sha256_file
from src.engine.output_utils import parse_model_output


def operation_counts(model: nn.Module, sample: torch.Tensor):
    counts={'convolution_mac':0,'linear_mac':0,'fft_butterfly_complexity_units':0,
            'fft_calls_estimated':0}
    handles=[]
    def record(module,inputs,output):
        if isinstance(module,(nn.Conv2d,nn.Conv3d,nn.ConvTranspose2d,nn.ConvTranspose3d)):
            if isinstance(output,(tuple,list)): output=output[0]
            if not isinstance(output,torch.Tensor):return
            kernel=math.prod(module.kernel_size)
            if isinstance(module,(nn.ConvTranspose2d,nn.ConvTranspose3d)):
                # Input spatial elements * out_channels/groups * kernel volume.
                mac=inputs[0].numel() * (module.out_channels//module.groups) * kernel
            else:
                mac=output.numel() * (module.in_channels//module.groups) * kernel
            counts['convolution_mac']+=int(mac)
        elif isinstance(module,nn.Linear):
            counts['linear_mac']+=int(output.numel()*module.in_features)
    def record_fft(module,inputs):
        # Two real FFTs (forward + inverse) acting on hidden channels.
        x=inputs[0]
        n=math.prod(x.shape[-2:]); channels=int(module.hidden_channels)
        batches=int(x.shape[0])
        counts['fft_calls_estimated']+=2
        counts['fft_butterfly_complexity_units']+=int(2*batches*channels*n*math.log2(max(2,n)))
    for module in model.modules():
        if isinstance(module,(nn.Conv2d,nn.Conv3d,nn.ConvTranspose2d,nn.ConvTranspose3d,nn.Linear)):
            handles.append(module.register_forward_hook(record))
        elif module.__class__.__name__=='FourierSpectralBottleneck':
            handles.append(module.register_forward_pre_hook(record_fft))
    try:
        with torch.inference_mode():
            _=parse_model_output(model(sample))
    finally:
        for handle in handles:handle.remove()
    counts['conv_linear_macs']=counts['convolution_mac']+counts['linear_mac']
    counts['conv_linear_flops_lower_bound']=2*counts['conv_linear_macs']
    counts['accounting_note']=(
        'Conv/linear MACs counted from actual tensor shapes; 2 FLOPs/MAC excludes '
        'bias, activation, pooling, interpolation, normalization and FFT. FFT '
        'butterfly complexity is a separate big-O proxy, NOT exact hardware FLOPs. '
        'These numbers are not a complete network FLOP count.')
    return counts


def benchmark(args):
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if args.repetitions<3 or args.batch_size<1 or args.warmup<0:
        raise ValueError('Require >=3 repetitions, positive batch size, nonnegative warmup')
    model,cfg,name=load_checkpoint(args.checkpoint,args.model,args.config,device)
    in_channels=int(cfg.get('model',{}).get('in_channels',3))
    sample=torch.randn(args.batch_size,in_channels,args.image_size,args.image_size,device=device)
    sync=lambda:torch.cuda.synchronize(device) if str(device).startswith('cuda') else None
    counts=operation_counts(model,sample)
    with torch.inference_mode():
        for _ in range(args.warmup):parse_model_output(model(sample))
        sync()
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
        latency=[]
        for _ in range(args.repetitions):
            sync();start=time.perf_counter()
            parse_model_output(model(sample))
            sync();latency.append((time.perf_counter()-start)*1000)
    vals=sorted(latency)
    q95=vals[max(0,math.ceil(.95*len(vals))-1)]
    gpu=str(device).startswith('cuda')
    result={'checkpoint_sha256':sha256_file(args.checkpoint),
            'model':name,'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if gpu else None,
            'pytorch_version':torch.__version__,'image_size':args.image_size,'batch_size':args.batch_size,
            'warmup':args.warmup,'repetitions':args.repetitions,
            'latency_ms_mean_batch':statistics.mean(vals),
            'latency_ms_median_batch':statistics.median(vals),
            'latency_ms_p95_batch':q95,
            'fps_images_per_second':1000*args.batch_size/statistics.mean(vals),
            'parameters_total':sum(p.numel() for p in model.parameters()),
            'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/1048576 if gpu else None,
            'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved(device)/1048576 if gpu else None,
            'operation_counts':counts}
    write_json(result,args.output)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--config');p.add_argument('--model')
    p.add_argument('--image-size',type=int,default=352);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--warmup',type=int,default=20);p.add_argument('--repetitions',type=int,default=100)
    p.add_argument('--device',default='auto');p.add_argument('--output',required=True)
    return p

if __name__=='__main__':
    result=benchmark(parser().parse_args())
    print(f"{result['model']}: {result['latency_ms_median_batch']:.3f} ms median; "
          f"{result['fps_images_per_second']:.2f} FPS")
