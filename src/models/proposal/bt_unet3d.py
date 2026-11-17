"""Volumetric (3D) extension of direct BT reconstruction, separate from 2D BT.

This is a new 3D research variant, not the submitted 2D checkpoint architecture.
All convolutions, normalization and FFTs operate on genuine 3D volumes.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _gn(c: int):
    groups=min(16,c)
    while c%groups:groups-=1
    return nn.GroupNorm(groups,c)


class DoubleConv3D(nn.Module):
    def __init__(self,cin:int,cout:int):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(cin,cout,3,padding=1,bias=False),_gn(cout),nn.GELU(),
                               nn.Conv3d(cout,cout,3,padding=1,bias=False),_gn(cout),nn.GELU())
    def forward(self,x):return self.net(x)


class SpectralBottleneck3D(nn.Module):
    """Direct 3D amplitude/phase modulation with independent one-sided FFT grid.

    Uses irfftn's real-output convention; unconstrained phase parameters on
    self-conjugate bins are not represented as an exact arbitrary complex multiplier.
    This matches the original 2D implementation convention and is stated plainly.
    """
    def __init__(self,channels:int,expansion:float=1.5,
                 base_grid:Sequence[int]=(8,8,8),amplitude_scale:float=1.,
                 phase_max:float=math.pi,dropout:float=.1):
        super().__init__()
        if len(base_grid)!=3 or min(base_grid)<1 or channels<1 or expansion<=0:
            raise ValueError('Positive channels, expansion and three base-grid dimensions required')
        hidden=max(channels,round(channels*expansion))
        d,h,w=map(int,base_grid)
        self.hidden_channels=int(hidden)
        self.amplitude_scale=float(amplitude_scale)
        self.phase_max=float(phase_max)
        self.pre=nn.Sequential(nn.Conv3d(channels,hidden,1,bias=False),_gn(hidden),nn.GELU())
        self.amplitude_logits=nn.Parameter(torch.zeros(1,hidden,d,h,w//2+1))
        self.phase_logits=nn.Parameter(torch.zeros(1,hidden,d,h,w//2+1))
        self.channel_mixer=nn.Conv3d(hidden,hidden,1,bias=False)
        self.dropout=nn.Dropout3d(dropout) if dropout else nn.Identity()
        self.post=nn.Conv3d(hidden,channels,1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.amplitude_logits,std=.001)
        nn.init.normal_(self.phase_logits,std=.001)
        with torch.no_grad():
            self.channel_mixer.weight.zero_()
            eye=torch.eye(self.hidden_channels,device=self.channel_mixer.weight.device)
            self.channel_mixer.weight[:,:,0,0,0].copy_(eye)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if x.ndim!=5:raise ValueError('Expect B,C,D,H,W features')
        z=self.pre(x)
        sizes=z.shape[-3:]
        spectrum=torch.fft.rfftn(z.float(),dim=(-3,-2,-1),norm='ortho')
        grid=spectrum.shape[-3:]
        amp=F.interpolate(self.amplitude_logits,size=grid,mode='trilinear',align_corners=False)
        phi=F.interpolate(self.phase_logits,size=grid,mode='trilinear',align_corners=False)
        gain=torch.exp(self.amplitude_scale*torch.tanh(amp))
        phase=self.phase_max*torch.tanh(phi)
        modulated=spectrum*torch.polar(gain,phase)
        mixed=torch.complex(self.channel_mixer(modulated.real),self.channel_mixer(modulated.imag))
        reconstructed=torch.fft.irfftn(mixed,s=sizes,dim=(-3,-2,-1),norm='ortho')
        return self.post(self.dropout(reconstructed.to(dtype=x.dtype)))


class UNet3D(nn.Module):
    """Small from-scratch 3D U-Net baseline with identical encoder/decoder to BT."""
    def __init__(self,in_channels=1,num_classes=1,channels=(16,32,64,128),
                 spectral=False,expansion=1.5,base_grid=(8,8,8)):
        super().__init__()
        if len(channels)<2:raise ValueError('At least two U-Net stages required')
        channels=tuple(int(c) for c in channels)
        self.encoder=nn.ModuleList([DoubleConv3D(in_channels if i==0 else channels[i-1],c)
                                   for i,c in enumerate(channels)])
        self.pool=nn.MaxPool3d(2)
        self.bottleneck=SpectralBottleneck3D(channels[-1],expansion,base_grid) if spectral else nn.Identity()
        self.decoder=nn.ModuleList([DoubleConv3D(channels[i+1]+channels[i],channels[i])
                                    for i in range(len(channels)-2,-1,-1)])
        self.seg_head=nn.Conv3d(channels[0],num_classes,1)
        self.spectral=bool(spectral)

    def forward(self,x):
        if x.ndim!=5:raise ValueError('Expected B,C,D,H,W')
        if min(x.shape[-3:])<2**(len(self.encoder)-1):
            raise ValueError('Input volume is too small for the 3D encoder depth')
        feats=[]
        for i,enc in enumerate(self.encoder):
            if i:x=self.pool(x)
            x=enc(x);feats.append(x)
        x=self.bottleneck(feats[-1])
        for layer,skip in zip(self.decoder,reversed(feats[:-1])):
            x=F.interpolate(x,size=skip.shape[-3:],mode='trilinear',align_corners=False)
            x=layer(torch.cat([x,skip],dim=1))
        return self.seg_head(x)


class BTUNet3D(UNet3D):
    def __init__(self,**kwargs):
        kwargs.pop('spectral',None)
        super().__init__(spectral=True,**kwargs)


__all__=['SpectralBottleneck3D','UNet3D','BTUNet3D']
