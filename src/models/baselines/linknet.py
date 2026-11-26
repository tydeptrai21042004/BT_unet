"""LinkNet (Chaurasia & Culurciello, VCIP 2017), ResNet-18, trained from scratch.
Paper-aligned PyTorch reimplementation, binary head adaptation, not author weights.
https://doi.org/10.1109/VCIP.2017.8305148
"""
import torch.nn as nn
import torch.nn.functional as F
from ..registry import register_model

class DecoderBlock(nn.Sequential):
    def __init__(self, cin, cout):
        mid=cin//4
        super().__init__(nn.Conv2d(cin,mid,1,bias=False),nn.BatchNorm2d(mid),nn.ReLU(inplace=True),
                         nn.ConvTranspose2d(mid,mid,3,stride=2,padding=1,output_padding=1,bias=False),nn.BatchNorm2d(mid),nn.ReLU(inplace=True),
                         nn.Conv2d(mid,cout,1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True))

@register_model('linknet')
class LinkNet(nn.Module):
    def __init__(self,in_channels=3,num_classes=1):
        super().__init__()
        from torchvision.models import resnet18
        enc=resnet18(weights=None)
        if in_channels!=3: enc.conv1=nn.Conv2d(in_channels,64,7,2,3,bias=False)
        self.stem=nn.Sequential(enc.conv1,enc.bn1,enc.relu,enc.maxpool)
        self.e1,self.e2,self.e3,self.e4=enc.layer1,enc.layer2,enc.layer3,enc.layer4
        self.d4,self.d3,self.d2=DecoderBlock(512,256),DecoderBlock(256,128),DecoderBlock(128,64)
        self.d1=DecoderBlock(64,64)
        self.head=nn.Sequential(nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1,bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True),nn.Conv2d(32,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True),nn.Conv2d(32,num_classes,1))
    @staticmethod
    def _add(x,y):
        if x.shape[-2:]!=y.shape[-2:]: x=F.interpolate(x,size=y.shape[-2:],mode='bilinear',align_corners=False)
        return x+y
    def forward(self,x):
        a=self.e1(self.stem(x)); b=self.e2(a); c=self.e3(b); d=self.e4(c)
        y=self._add(self.d4(d),c); y=self._add(self.d3(y),b); y=self._add(self.d2(y),a)
        y=self.head(self.d1(y))
        return F.interpolate(y,size=x.shape[-2:],mode='bilinear',align_corners=False) if y.shape[-2:]!=x.shape[-2:] else y
