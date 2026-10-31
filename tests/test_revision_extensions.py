"""Regression coverage for reviewer-oriented controls and leakage safeguards."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.models import build_model
from src.analysis.boundary_metrics import measure_binary
from src.analysis.corruptions import SUPPORTED, corrupt
from scripts.train_one import build_scheduler, load_config_from_args


@pytest.mark.parametrize('name',[
    'dapr_unet','dapr_matched_residual','dapr_matched_residual_zero_init','dapr_matched_difference',
    'dapr_amplitude_only','dapr_phase_only','dapr_mixing_only',
])
def test_residual_models_produce_correct_shapes(name):
    torch.manual_seed(4)
    model=build_model(name,config={'channels':(8,16,32,64,128), 'fourier_init_hw':(2,2), 'fourier_dropout':0.0}).eval()
    out=model(torch.randn(1,3,32,32))
    assert out.shape==(1,1,32,32)
    assert torch.isfinite(out).all()
    assert model.fourier_bottleneck.residual == (name in {'dapr_matched_residual','dapr_matched_residual_zero_init','dapr_matched_difference'})
    if name=='dapr_matched_residual':
        assert model.fourier_bottleneck.residual_signal=='reconstructed'
        assert not model.fourier_bottleneck.zero_init_output


def test_scheduler_horizon_following_cli_override():
    from argparse import Namespace
    args=Namespace(model='dapr_unet',dataset='kvasir_seg',config='configs/fair/dapr_unet.yaml',
        data_root=None,image_size=None,batch_size=None,epochs=50,lr=None,device=None,
        output_root=None,num_workers=None,seed=None,resume=None,
        train_datasets=None,val_dataset=None,train_split_file=None)
    cfg=load_config_from_args(args)
    assert cfg['train']['t_max']==cfg['train']['epochs']==50
    model=torch.nn.Linear(2,1)
    scheduler=build_scheduler(torch.optim.Adam(model.parameters()),cfg)
    assert scheduler.T_max==50


def test_boundary_metrics_empty_and_exact():
    zero=np.zeros((16,16))
    perfect=measure_binary(zero,zero)
    assert perfect['dice']==1 and perfect['boundary_f1']==1 and perfect['hd95_pixels']==0
    target=zero.copy();target[3:11,4:12]=1
    equal=measure_binary(target,target)
    assert equal['dice']==1 and equal['boundary_f1']==1
    assert math.isnan(measure_binary(zero,target)['hd95_pixels'])


@pytest.mark.parametrize('kind',SUPPORTED)
def test_corruptions_reproducible_and_preserve_dimensions(kind):
    image=Image.fromarray(np.full((32,40,3),125,dtype=np.uint8))
    a=corrupt(image,kind,2,seed=9,sample_id='patient_a')
    b=corrupt(image,kind,2,seed=9,sample_id='patient_a')
    assert a.size==image.size
    assert np.array_equal(np.asarray(a),np.asarray(b))


def test_external_baseline_requires_valid_module():
    with pytest.raises((ModuleNotFoundError,ImportError)):
        build_model('external_spectral', config={'module_path':'nonexistent_zzzzz_module', 'class_name':'Net'})


def test_nested_low_data_manifest_preserves_original_splits(tmp_path):
    rev_dir=Path(__file__).resolve().parents[1]/'scripts/revision'
    sys.path.insert(0,str(rev_dir))
    from make_low_data_splits import make_fractions
    for split,values in [('train',[f'x{i}' for i in range(20)]),('val',['v0']),('test',['t0'])]:
        loc=tmp_path/'splits'/'kvasir_seg'/f'{split}.txt';loc.parent.mkdir(parents=True,exist_ok=True)
        loc.write_text('\n'.join(values)+'\n')
    records=make_fractions(tmp_path,'kvasir_seg',[.1,.5,1.],42,tmp_path/'revision')
    a=set(Path(records['0.1']['path']).read_text().splitlines())
    b=set(Path(records['0.5']['path']).read_text().splitlines())
    assert a<b and len(a)==2 and len(b)==10
    assert (tmp_path/'splits/kvasir_seg/test.txt').read_text().strip()=='t0'
