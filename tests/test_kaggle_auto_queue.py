"""Offline AUTO queue regression tests. No network, dataset download or GPU required."""
import csv
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/kaggle/multi_t4_runner.py'


def auto(tmp_path, monkeypatch, start=1):
    monkeypatch.setattr(sys,'argv',['multi_t4_runner.py','--scope','AUTO','--one-based',
                                  '--start-index',str(start),'--bundle-session'])
    monkeypatch.setenv('BT_KAGGLE_WORKDIR',str(tmp_path))
    spec=importlib.util.spec_from_file_location('runner_auto_'+str(start),SCRIPT)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_auto_plan_153_no_private_google_drive(tmp_path):
    env=os.environ.copy()
    env['BT_KAGGLE_WORKDIR']=str(tmp_path)
    p=subprocess.run([sys.executable,str(SCRIPT),'--scope','AUTO','--one-based',
        '--start-index','1','--plan-only'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert '001/153' in p.stdout and '153/153' in p.stdout
    rows=list(csv.DictReader((tmp_path/'BT_UNET_PLAN_AUTO.csv').open()))
    assert len(rows)==153
    assert [int(x['experiment_number']) for x in rows]==list(range(1,154))
    assert len({x['id'] for x in rows})==153
    assert set(x['dataset'] for x in rows)=={'kvasir_seg','kvasir_instrument','isbi2012','hyper_kvasir_seg'}
    assert all(x['needs_extra_sources']=='False' for x in rows)
    assert all(x['type']!='cross' for x in rows)


def test_auto_prep_calls_public_source_no_pair_audit(tmp_path, monkeypatch):
    r=auto(tmp_path,monkeypatch)
    r._activate(r.QUEUE[0])
    calls=[]
    def fake_cmd(cmd,**kwargs):
        calls.append(list(map(str,cmd)))
        if 'make_splits.py' in cmd:
            splits=r.DATA/'splits'/r.SPEC['dataset']
            splits.mkdir(parents=True,exist_ok=True)
            for name in ('train','val','test'):(splits/(name+'.txt')).write_text('sample_1\n')
    monkeypatch.setattr(r,'run_cmd',fake_cmd)
    r.setup_datasets()
    assert len(calls)==2
    assert 'scripts/prepare_dataset.py' in calls[0]
    assert not {'--source-dir','--zip-path','--download-url'}.intersection(calls[0])
    assert 'scripts/make_splits.py' in calls[1]
    assert r.RUN_META['dataset_provenance']['kvasir_seg']['source']=='public_https_direct_auto'
    assert 'dataset_content_sha256' not in r.RUN_META
    assert 'kvasir_seg' in r.PREPARED


def test_auto_resume_and_no_empty_bundle(tmp_path,monkeypatch):
    r=auto(tmp_path,monkeypatch,start=1)
    r.export_plan()
    r.SESSION.mkdir(parents=True,exist_ok=True)
    status={'scope':'AUTO','next_index':0,'completed_ids':[],
            'session_archives':[], 'git_commit':'f'*40, 'stop_reason':'failed'}
    r._save_queue(status)
    r.bundle_session(status)
    assert (tmp_path/'BT_UNET_NEXT_EXPERIMENT_AUTO.txt').read_text().strip()=='1'
    assert not list(tmp_path.glob('BT_UNET_SESSION_AUTO_*.zip'))


def test_auto_completed_bundle_and_merge(tmp_path,monkeypatch):
    r=auto(tmp_path,monkeypatch,start=1)
    r.export_plan()
    r.SESSION.mkdir(parents=True,exist_ok=True)
    job=tmp_path/'BT_UNET_RESULTS__job1.zip'
    with zipfile.ZipFile(job,'w') as z:z.writestr('test.json','{}')
    status={'scope':'AUTO','next_index':1,'completed_ids':[r.QUEUE[0]['id']],
            'session_archives':[str(job)], 'git_commit':'f'*40, 'stop_reason':'budget'}
    r._save_queue(status);r.bundle_session(status)
    zipped=tmp_path/'BT_UNET_SESSION_AUTO_0001_TO_0001.zip'
    assert zipped.is_file()
    assert (tmp_path/'BT_UNET_NEXT_EXPERIMENT_AUTO.txt').read_text().strip()=='2'
    with zipfile.ZipFile(zipped) as z:
        manifest=json.loads(z.read('session_bundle_manifest.json'))
        assert manifest['total_experiments']==153
        assert manifest['next_experiment_number']==2
    from scripts.kaggle.merge_sessions import collect,merge
    packages,nxt=collect(tmp_path,scope='AUTO')
    assert len(packages)==1 and nxt==2
    try:
        merge(tmp_path,tmp_path/'final.zip',require_complete=True,scope='AUTO')
    except ValueError as exc:
        assert 'Expected 153' in str(exc)
    else:
        raise AssertionError('Incomplete results should not be declared complete')
