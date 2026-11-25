import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scripts.kaggle import auto_acquire as ac
from src.engine import recovery
from src.engine.trainer import Trainer


def test_validated_pair_count_and_source_fallback(tmp_path, monkeypatch):
    monkeypatch.setitem(ac.EXPECTED,'kvasir_instrument',2)
    source=tmp_path/'source'; (source/'images').mkdir(parents=True); (source/'masks').mkdir()
    for i in range(2):
        Image.new('RGB',(8,8)).save(source/'images'/f'{i}.png')
        Image.new('L',(8,8),255).save(source/'masks'/f'{i}.png')
    import sys, types
    def fail(_): raise RuntimeError('simulated Kaggle outage')
    monkeypatch.setitem(sys.modules,'kagglehub',types.SimpleNamespace(dataset_download=fail))
    import zipfile
    z=tmp_path/'source.zip'
    with zipfile.ZipFile(z,'w') as f:
        for p in source.rglob('*.png'): f.write(p,p.relative_to(source))
    monkeypatch.setattr(ac,'_maybe_download',lambda *a,**k:z)
    receipt=ac.acquire('kvasir_instrument',tmp_path/'cache')
    assert receipt['pairs']==2 and receipt['source']=='https_zip' and receipt['failed_sources']
    assert ac.acquire('kvasir_instrument',tmp_path/'cache')==receipt


def test_full_state_resume_matches_uninterrupted(tmp_path):
    def make():
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        model=torch.nn.Conv2d(1,1,1)
        optim=torch.optim.Adam(model.parameters(),lr=.02)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=3)
        t=Trainer(model,optim,torch.nn.BCEWithLogitsLoss(),device='cpu',scheduler=sched,mixed_precision=False)
        def train(loader,epoch):
            optim.zero_grad(); loss=model(torch.rand(1,1,2,2)).square().mean();loss.backward();optim.step();sched.step(); return {'loss':loss.item()}
        t.train_one_epoch=train
        return t
    a=make(); full=a.fit(None,epochs=3,monitor='loss',monitor_mode='min')
    b=make(); checkpoint=tmp_path/'recovery.pt'; b.fit(None,epochs=1,monitor='loss',monitor_mode='min',recovery_path=checkpoint,recovery_identity='same')
    c=make(); resumed=c.fit(None,epochs=3,monitor='loss',monitor_mode='min',recovery_path=checkpoint,recovery_identity='same')
    assert resumed==full
    assert all(torch.equal(x,y) for x,y in zip(a.model.parameters(),c.model.parameters()))
    assert c.scheduler.state_dict()==a.scheduler.state_dict()
