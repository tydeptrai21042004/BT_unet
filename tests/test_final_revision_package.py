"""Regression tests for reviewer package additions, without external data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import build_model
from src.models.proposal.fourier_unet import FourierSpectralBottleneck
from src.models.official.strict_external import _resolve_primary, verify_source
from src.analysis.boundary_metrics import measure_binary
from scripts.revision.plot_convergence_multiseed import aggregate_histories
from scripts.revision.run_full_revision import build_plan
from scripts.revision.build_manuscript_tables import summarize,tex_table


def compact(name):
    return build_model(name,config={'channels':(4,8,16),
        'fourier_expansion':1., 'fourier_init_hw':(4,4),
        'fourier_dropout':0.0,'fourier_block_norm':'gn'}).eval()


@pytest.mark.parametrize('name',["dapr_identity_spectral_control", "dapr_spatial_channel_mix_control"])
def test_new_controls_forward_backprop(name):
    torch.manual_seed(8)
    model=compact(name).train()
    x=torch.randn(2,3,32,32)
    output=model(x)
    assert output.shape==(2,1,32,32)
    output.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_fft_mixer_commutes_with_real_spatial_1x1():
    torch.manual_seed(9)
    a=FourierSpectralBottleneck(channels=8,expansion=1.,dropout=0.,
             init_hw=(8,8),residual=False,zero_init_output=False,channel_mix_location='frequency')
    b=FourierSpectralBottleneck(channels=8,expansion=1.,dropout=0.,
             init_hw=(8,8),residual=False,zero_init_output=False,channel_mix_location='spatial')
    b.load_state_dict(a.state_dict())
    a.eval();b.eval()
    for shape in [(2,8,8,8),(1,8,10,12)]:
        x=torch.randn(*shape)
        torch.testing.assert_close(a(x),b(x),rtol=1e-4,atol=2e-5)


def test_identity_spectral_control_is_valid():
    model=FourierSpectralBottleneck(8,expansion=1.,init_hw=(8,8),
        use_amplitude=False,use_phase=False,use_channel_mixing=False,
        residual=False,zero_init_output=False)
    assert torch.isfinite(model(torch.randn(1,8,8,8))).all()


def test_revision_run_plan_unique_training_seeds(tmp_path):
    args=argparse.Namespace(seeds='12,34',groups='factorial,residual,convergence',
        datasets='kvasir_seg',epochs=1,image_size=64,output_root=str(tmp_path),
        data_root=str(tmp_path/'data'),batch_size=2,eval_batch_size=2,num_workers=0,
        device='cpu',external_configs='',split='test',save_predictions=False)
    jobs=build_plan(args)
    assert len(jobs)==(8+7+4)*2
    assert len(set(j['id'] for j in jobs))==len(jobs)
    assert all(j['config']['train']['t_max']==1 for j in jobs)
    assert all(j['config']['train']['seed']==j['seed'] for j in jobs)
    assert any(j['config']['model'].get('fourier_alpha')==1.0 for j in jobs)


def test_official_adapter_rejects_unaudited_or_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_source(tmp_path/'missing','deadbeef',tmp_path/'audit.json')
    with pytest.raises(TypeError):
        _resolve_primary((torch.zeros(1,1,8,8),),None,None)
    assert _resolve_primary((torch.zeros(1,1,8,8),),None,0).shape==(1,1,8,8)


def test_complete_misses_are_counted():
    target=np.zeros((16,16));target[2:7,2:7]=1
    out=measure_binary(np.zeros_like(target),target)
    assert out['complete_miss']==1 and out['distance_defined']==0


def test_multiseed_mean_curve_from_only_recorded_epochs(tmp_path):
    files=[]
    for n in [1,2]:
        f=tmp_path/f'{n}.csv'
        with f.open('w',newline='') as out:
            w=csv.DictWriter(out,fieldnames=['epoch','val/dice']);w.writeheader()
            w.writerows([{'epoch':1,'val/dice':n/10},{'epoch':2,'val/dice':(n+1)/10}])
        files.append(f'Model={f}')
    rows=aggregate_histories(files)
    assert len(rows)==2 and rows[0]['n_seeds']==2
    assert np.isclose(rows[0]['mean'],.15)


def test_aggregation_rejects_unfinished_runs(tmp_path):
    plan={'jobs':[{'dataset':'kvasir_seg','group':'convergence','variant':'unet',
                   'model':'unet','seed':42,'metrics_dir':str(tmp_path/'missing')}]}
    f=tmp_path/'plan.json';f.write_text(json.dumps(plan))
    with pytest.raises(FileNotFoundError):
        summarize(f,strict=True)
    assert 'tabular' in tex_table([{'dataset':'kvasir_seg','group':'convergence','variant':'unet',
                                  'n_seeds':1,'dice_mean':.7,'dice_std':float('nan')}])


def test_strict_source_adapter_loads_only_pinned_audited_code(tmp_path):
    import subprocess
    upstream=tmp_path/'upstream';upstream.mkdir()
    (upstream/'toy_official_stub.py').write_text('''import torch
from torch import nn
class TinyNet(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.head=nn.Conv2d(channels,1,1)
    def forward(self,x):
        return {'prediction':self.head(x)}
''')
    subprocess.run(['git','init','-q',str(upstream)],check=True)
    subprocess.run(['git','-C',str(upstream),'add','.'],check=True)
    subprocess.run(['git','-C',str(upstream),'-c','user.name=Test',
                    '-c','user.email=local@example.org','commit','-qm','fixture'],check=True)
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    audit=tmp_path/'reviewed.json'
    audit.write_text(json.dumps({'status':'verified','official_repository':'fixture://not-a-paper',
         'paper_reference':'fixture only','commit':commit,
         'component_checks':{'encoder':'toy_official_stub.py::TinyNet',
                             'decoder':'toy_official_stub.py::TinyNet',
                             'output_head':'toy_official_stub.py::TinyNet'}}))
    model=build_model('verified_external_spectral',config={
        'source_dir':str(upstream),'expected_commit':commit,'audit_file':str(audit),
        'module_path':'toy_official_stub','entrypoint':'TinyNet','model_kwargs':{'channels':3},
        'output_key':'prediction','output_type':'logits'})
    assert model(torch.randn(2,3,16,16)).shape==(2,1,16,16)
    with pytest.raises(ValueError):
        build_model('verified_external_spectral',config={
            'source_dir':str(upstream),'expected_commit':'deadbeef',
            'audit_file':str(audit),'module_path':'toy_official_stub','entrypoint':'TinyNet'})
