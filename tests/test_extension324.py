import csv, json, os, sys, zipfile, importlib.util
from pathlib import Path
from collections import Counter
import pytest
from scripts.kaggle.extension324 import proposal_family
from scripts.kaggle.reuse_completed import import_rows
from scripts.kaggle.results_export import MAIN

ROOT=Path(__file__).resolve().parents[1]

def get_runner(tmp_path,monkeypatch,revised):
    monkeypatch.setenv('BT_KAGGLE_WORKDIR',str(tmp_path))
    monkeypatch.setattr(sys,'argv',['multi_t4_runner.py','--scope','AUTO','--plan-only','--extend-300']+(['--rerun-proposal'] if revised else []))
    spec=importlib.util.spec_from_file_location('runner324',ROOT/'scripts/kaggle/multi_t4_runner.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    return r

def test_revised_queue_and_actual_configs(tmp_path,monkeypatch):
    old=get_runner(tmp_path,monkeypatch,False)
    new=get_runner(tmp_path,monkeypatch,True)
    assert len(new.QUEUE)==324
    assert len({j['id'] for j in new.QUEUE})==324
    assert sum(j.get('fourier_hermitian_constraint',False) for j in new.QUEUE)==105
    assert sum(j.get('fourier_hermitian_constraint',False) for j in new.QUEUE[:165])==93
    assert Counter(j['dataset'] for j in new.QUEUE if j['model']=='linknet')=={d:3 for d in new.AUTO_DATASETS}
    for before,after in zip(old.QUEUE[:288],new.QUEUE[:288]):
        if proposal_family(before):
            assert after==dict(before,id='h2__'+before['id'],fourier_hermitian_constraint=True)
        else: assert before==after
    # Resolve all actual YAMLs (not just the manifest) for the 105 revised jobs.
    low=new.SESSION/'common_lowdata_splits';low.mkdir(parents=True)
    for fraction in (10,25,50,100): (low/f'train_{fraction:03d}pct_seed42.txt').write_text('fake_sample\n')
    for j in new.QUEUE:
        if j.get('fourier_hermitian_constraint') or j['model']=='linknet' or j.get('group') in {'expanded_baselines','matched_hyper_baselines'}:
            new._activate(j)
            _,_,_,cfg,_=new.make_job()
            assert cfg['model'].get('fourier_hermitian_constraint',False)==j.get('fourier_hermitian_constraint',False)
            assert cfg['train']['epochs']==j['epochs']
            assert cfg['train']['seed']==j['seed']
    new.export_plan()
    assert (tmp_path/'BT_UNET_PLAN_AUTO.csv').read_bytes()==(ROOT/'BT_UNET_PLAN_AUTO_324.csv').read_bytes()

def write_archive(path, plan_path, specs, indices):
    with plan_path.open(newline='') as f: plan=list(csv.DictReader(f))
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('session_bundle_manifest.json',json.dumps({'scope':'AUTO','total_experiments':len(plan),'completed_experiment_numbers':indices,'archives':[{'name':f'{n:04d}','format':'compact_metrics'} for n in indices]}))
        z.write(plan_path,'BT_UNET_PLAN_AUTO.csv')
        for n in indices:
            p=plan[n-1];spec=specs[n-1];prefix=f'completed_jobs/{n:04d}/'
            row={'experiment_number':n,'experiment_id':p['id'],'dataset':p['dataset'],'model':p['model'],'variant':p['variant'],'seed':p['seed'],'epochs':p['epochs'],'status':'complete',**{k:'0.500000' for k in MAIN}}
            z.writestr(prefix+'paper_row.json',json.dumps(row));z.writestr(prefix+'provenance.json',json.dumps({'experiment':spec,'checkpoint_sha256':'a'*64,'test_split':'test'}))

def test_archive_reuse_never_uses_old_proposal(tmp_path,monkeypatch):
    old=get_runner(tmp_path,monkeypatch,False);new=get_runner(tmp_path,monkeypatch,True)
    write_archive(tmp_path/'BT_UNET_SESSION_AUTO_old.zip',ROOT/'BT_UNET_PLAN_AUTO_300.csv',old.QUEUE,[1,4,13,37,118,154,274,289])
    write_archive(tmp_path/'BT_UNET_SESSION_AUTO_H2_new.zip',ROOT/'BT_UNET_PLAN_AUTO_324.csv',new.QUEUE,[4,13])
    new.export_plan()
    actual=import_rows(tmp_path,new.PLAN_CSV,tmp_path/'compact',new.QUEUE)
    assert actual=={1,4,13,274}
    for n in (4,13): assert json.loads((tmp_path/'compact'/f'{n:04d}'/'paper_row.json').read_text())['experiment_id'].startswith('h2__')
    from scripts.kaggle.merge_paper_metrics import merge_paper_results
    rows=merge_paper_results(tmp_path,tmp_path/'merged')
    assert len(rows)==324 and rows[3]['status']=='complete' and rows[36]['status']=='pending'
