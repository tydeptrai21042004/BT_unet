"""Regression checks for the reviewer-driven code additions.

Synthetic tiny arrays are unit-test fixtures, NEVER evidence of medical accuracy.
"""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.revision.audit_splits import audit,read_cluster_map
from scripts.revision.run_full_revision import build_plan
from scripts.revision.plot_qualitative import boundary_crop
from scripts.revision.benchmark_inference import operation_counts
from scripts.revision.statistical_comparison import holm
from scripts.revision.run_volumetric import predict_volume, segmentation_metrics, build_3d
from src.datasets.medical_volume_dataset import MedicalVolumeDataset,audit_patient_splits
from src.models.proposal.bt_unet3d import BTUNet3D,UNet3D,SpectralBottleneck3D


def _volume_fixture(path, split, patient, sample):
    root=path/'data';root.mkdir(exist_ok=True)
    x=np.zeros((16,16,16),np.float32)
    x[3:10,3:11,4:12]=1
    y=np.zeros_like(x);y[4:11,4:11,4:12]=1
    np.save(root/f'{sample}_image.npy',x)
    np.save(root/f'{sample}_mask.npy',y)
    folder=path/'splits';folder.mkdir(exist_ok=True)
    with (folder/f'{split}.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=['sample_id','patient_id','image','mask'])
        w.writeheader();w.writerow({'sample_id':sample,'patient_id':patient,
                                    'image':f'{sample}_image.npy','mask':f'{sample}_mask.npy'})
    return root,folder


def test_3d_bottleneck_shapes_and_gradients():
    torch.set_num_threads(1)
    for cls in (BTUNet3D,UNet3D):
        model=cls(channels=(4,8,16))
        data=torch.randn(1,1,16,16,16)
        out=model(data)
        assert out.shape==(1,1,16,16,16)
        out.mean().backward()
        assert torch.isfinite(out).all()
        assert any(p.grad is not None for p in model.parameters())


def test_3d_sliding_window_complete_coverage():
    torch.set_num_threads(1)
    model=BTUNet3D(channels=(4,8,16)).eval()
    x=torch.randn(1,1,19,23,18)
    probs=predict_volume(model,x,patch=(16,16,16),overlap=.5)
    assert probs.shape==(1,1,19,23,18)
    assert torch.isfinite(probs).all()
    assert probs.min()>=0 and probs.max()<=1


def test_3d_patient_group_separation(tmp_path):
    root,manifests=_volume_fixture(tmp_path,'train','patientA','sampleA')
    _volume_fixture(tmp_path,'val','patientB','sampleB')
    _volume_fixture(tmp_path,'test','patientC','sampleC')
    summary=audit_patient_splits(*(manifests/f'{s}.csv' for s in ('train','val','test')))
    assert summary['train']['patients']==1
    ds=MedicalVolumeDataset(root,manifests/'train.csv',patch_shape=(16,16,16),train=True)
    assert ds[0]['image'].shape==(1,16,16,16)
    assert ds[0]['mask'].shape==(1,16,16,16)
    (manifests/'test.csv').write_text((manifests/'train.csv').read_text())
    with pytest.raises(ValueError,match='Patient-level leakage'):
        audit_patient_splits(*(manifests/f'{s}.csv' for s in ('train','val','test')))


def test_volumetric_spacing_and_metrics():
    pred=np.zeros((8,8,8),np.float32);gt=pred.copy()
    gt[2:5,2:5,2:5]=1.;pred[2:5,2:5,2:5]=1.
    r=segmentation_metrics(pred,gt,(2.,1.,1.))
    assert r['dice']==1. and r['hd95_mm']==0.


def test_boundary_crop_deterministic():
    target=np.zeros((128,160),bool);target[20:65,45:80]=True
    a=boundary_crop(target,40)
    assert a==boundary_crop(target,40)
    assert a[2]-a[0]==40 and a[3]-a[1]==40


def test_run_plan_primary_and_residual_have_valid_configs(tmp_path):
    args=SimpleNamespace(seeds='42,1337',groups='primary,residual,factorial',datasets='kvasir_seg',
          epochs=2,image_size=64,output_root=str(tmp_path/'out'),data_root=str(tmp_path/'data'),
          batch_size=1,eval_batch_size=1,num_workers=0,device='cpu',external_configs='',
          split='test',save_predictions=False)
    jobs=build_plan(args)
    assert len(jobs)==(2+7+8)*2
    for job in jobs:
        assert Path(job['source_config']).is_file()
        assert job['config']['train']['seed'] in (42,1337)
    item=[j for j in jobs if 'matched_difference' in j['model']][0]
    assert item['source_config'].endswith('dapr_matched_difference.yaml')


def test_2d_split_cluster_leakage_is_rejected(tmp_path):
    splits=tmp_path/'splits'/'kvasir_seg';splits.mkdir(parents=True)
    for sp,ids in [('train',['a']),('val',['b']),('test',['c'])]:
        (splits/f'{sp}.txt').write_text('\n'.join(ids)+'\n')
    with (splits/'clusters.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=['sample_id','cluster_id']);w.writeheader()
        for sid,pid in [('a','patient1'),('b','patient2'),('c','patient1')]:
            w.writerow({'sample_id':sid,'cluster_id':pid})
    with pytest.raises(ValueError,match='Data leakage'):
        audit(tmp_path,['kvasir_seg'],compare_content=False)
    (splits/'clusters.csv').unlink()
    report=audit(tmp_path,['kvasir_seg'])
    assert report['patient_video_integrity_verified']['kvasir_seg'] is False


def test_benchmark_counts_separately_from_fft():
    torch.set_num_threads(1)
    model=BTUNet3D(channels=(4,8,16)).eval()
    counts=operation_counts(model,torch.randn(1,1,16,16,16))
    assert counts['conv_linear_macs']>0
    assert 'FFT' in counts['accounting_note']
    assert counts['fft_calls_estimated']==0 # 3D model uses distinct class, not 2D bottleneck hook


def test_holm_wilcoxon_outputs():
    assert holm([.01,.03,.02])==pytest.approx([.03,.04,.04])
