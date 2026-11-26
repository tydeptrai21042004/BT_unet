import csv, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from scripts.kaggle.extension300 import append_jobs
from src.models import build_model
from src.models.proposal.fourier_unet import FourierSpectralBottleneck
from src.analysis.boundary_metrics import measure_binary
from src.analysis.extended_metrics import probability_metrics, confusion_metrics

def test_queue_prefix(tmp_path):
    root=Path(__file__).resolve().parents[1]
    plans=[]
    for extra in ([],['--extend-300']):
        subprocess.run([sys.executable,str(root/'scripts/kaggle/multi_t4_runner.py'),'--scope','AUTO','--plan-only',*extra],env=dict(os.environ,BT_KAGGLE_WORKDIR=str(tmp_path)),check=True,stdout=subprocess.DEVNULL)
        with (tmp_path/'BT_UNET_PLAN_AUTO.csv').open() as f: plans.append(list(csv.DictReader(f)))
    assert len(plans[1])==300 and plans[1][:165]==plans[0]
    assert len({r['id'] for r in plans[1]})==300

def test_linknet_forward_backward():
    torch.set_num_threads(2)
    model=build_model('linknet',{'in_channels':3,'num_classes':1})
    x=torch.randn(2,3,64,64)
    y=model(x); assert y.shape==(2,1,64,64)
    y.mean().backward(); assert model.head[-1].weight.grad is not None
    model.eval()
    with torch.no_grad(): assert model(torch.randn(1,3,352,352)).shape==(1,1,352,352)

def test_hermitian_even_odd_and_legacy():
    for h,w in [(8,8),(7,9),(8,9),(7,8)]:
        torch.manual_seed(2)
        b=FourierSpectralBottleneck(4,init_hw=(h,w),residual=False,zero_init_output=False,hermitian_constraint=True)
        with torch.no_grad(): b.phase_logits.normal_();b.amplitude_logits.normal_()
        x=torch.randn(2,4,h,w);s=torch.fft.rfft2(x,norm='ortho')
        z=torch.fft.rfft2(b.pre_act(b.pre_norm(b.pre(x))),norm='ortho')
        a,p=b._response(z,w);y=b._mix(z*torch.polar(a,p))
        y2=torch.fft.rfft2(torch.fft.irfft2(y,s=(h,w),norm='ortho'),norm='ortho')
        assert torch.allclose(y,y2,atol=3e-6,rtol=1e-5)
        out=b(x);out.square().mean().backward();assert torch.isfinite(b.phase_logits.grad).all()
        legacy=FourierSpectralBottleneck(4,init_hw=(h,w),residual=False,zero_init_output=False)
        legacy.load_state_dict(b.state_dict(),strict=True)

def test_extended_metrics():
    y=np.array([[True,False],[True,False]])
    p=y.astype(float)
    assert measure_binary(p,y)['surface_dice']==1
    m=probability_metrics(p,y);assert m['brier']==0 and m['ece']==0 and m['auprc']==1
    assert probability_metrics(np.full((2,2),.5),y)['auprc']==.5
    assert confusion_metrics(2,0,2,0)['mcc']==1
    assert np.isnan(confusion_metrics(0,0,4,0)['mcc'])

def test_reuse_legacy_archive(tmp_path):
    import io,json,zipfile,hashlib
    from scripts.kaggle.reuse_completed import import_rows
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'scripts/kaggle/multi_t4_runner.py'),'--scope','AUTO','--plan-only'],env=dict(os.environ,BT_KAGGLE_WORKDIR=str(tmp_path)),check=True,stdout=subprocess.DEVNULL)
    original=(tmp_path/'BT_UNET_PLAN_AUTO.csv').read_text()
    plan=list(csv.DictReader(io.StringIO(original)))
    spec=plan[0]
    row={'experiment_number':1,'experiment_id':spec['id'],'dataset':spec['dataset'],'model':spec['model'],'variant':spec['variant'],'seed':spec['seed'],'epochs':spec['epochs'],'status':'complete',**{k:'.5' for k in ('dice','iou','precision','recall','mae','loss')}}
    zpath=tmp_path/'BT_UNET_SESSION_AUTO_0001_TO_0001.zip'
    with zipfile.ZipFile(zpath,'w') as z:
        z.writestr('session_bundle_manifest.json',json.dumps({'scope':'AUTO','completed_experiment_numbers':[1],'archives':[{'name':'0001','format':'compact_metrics'}]}))
        z.writestr('BT_UNET_PLAN_AUTO.csv',original)
        z.writestr('completed_jobs/0001/paper_row.json',json.dumps(row))
        z.writestr('completed_jobs/0001/provenance.json',json.dumps({'experiment':{'id':spec['id']},'checkpoint_sha256':'a'*64,'test_split':'test','test_sample_count':1}))
        z.writestr('completed_jobs/0001/evaluations/'+spec['dataset']+'/per_image.csv','complete_miss,empty_ground_truth,empty_prediction,distance_defined\n1,0,1,0\n')
    subprocess.run([sys.executable,str(root/'scripts/kaggle/multi_t4_runner.py'),'--scope','AUTO','--extend-300','--plan-only'],env=dict(os.environ,BT_KAGGLE_WORKDIR=str(tmp_path)),check=True,stdout=subprocess.DEVNULL)
    assert import_rows(tmp_path,tmp_path/'BT_UNET_PLAN_AUTO.csv',tmp_path/'compact')=={1}
    saved=json.loads((tmp_path/'compact/0001/paper_row.json').read_text())
    assert saved['dice']=='.5' and saved['complete_miss_rate']=='1.000000'
