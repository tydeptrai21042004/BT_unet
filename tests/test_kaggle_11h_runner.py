"""CPU-only tests: no network, GPU, real training or invented paper metrics."""
from __future__ import annotations
import importlib.util
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/kaggle/multi_t4_runner.py'


def load_runner(tmp_path,monkeypatch, start=1):
    monkeypatch.setattr(sys,'argv',['multi_t4_runner.py','--scope','ALL','--one-based',
                                    '--start-index',str(start),'--bundle-session'])
    monkeypatch.setenv('BT_KAGGLE_WORKDIR',str(tmp_path))
    name='runner_test_'+str(start)
    spec=importlib.util.spec_from_file_location(name,SCRIPT)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_one_based_plan_has_261_unique_jobs(tmp_path):
    env=os.environ.copy();env['BT_KAGGLE_WORKDIR']=str(tmp_path)
    run=subprocess.run([sys.executable,str(SCRIPT),'--scope','ALL','--one-based',
                        '--start-index','1','--plan-only'],capture_output=True,text=True,
                       cwd=ROOT,env=env)
    assert run.returncode==0,run.stderr
    assert '001/261' in run.stdout
    import csv
    rows=list(csv.DictReader((tmp_path/'BT_UNET_PLAN_ALL.csv').open()))
    assert len(rows)==261
    assert rows[0]['experiment_number']=='1'
    assert rows[-1]['experiment_number']=='261'
    assert len({x['id'] for x in rows})==261
    assert rows[0]['dataset']=='kvasir_seg'


def test_residual_control_not_mislabeled_unknown(tmp_path,monkeypatch):
    runner=load_runner(tmp_path,monkeypatch)
    job=next(x for x in runner.QUEUE if x['type']=='revision' and x['group']=='residual'
             and x['model'].startswith('additive_'))
    runner.SPEC=job
    runner.RUN_META={}
    runner.baseline_gate()
    assert runner.RUN_META['baseline_fidelity']=='BT_unet_repository_reviewer_control'


def test_resume_counter_and_single_bundle(tmp_path,monkeypatch):
    runner=load_runner(tmp_path,monkeypatch)
    runner.export_plan()
    runner.SESSION.mkdir(parents=True,exist_ok=True)
    archive=tmp_path/'BT_UNET_RESULTS__test_job.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('result.json','{"dice": 0.5}')
    status={'scope':'ALL','next_index':1,'completed_ids':['test_job'],
            'session_archives':[str(archive)],'stop_reason':'more_jobs_remaining',
            'git_commit':'a'*40}
    runner._save_queue(status)
    assert (tmp_path/'BT_UNET_NEXT_EXPERIMENT_ALL.txt').read_text().strip()=='2'
    assert json.loads((tmp_path/'BT_UNET_NEXT_ALL.json').read_text())['next_experiment_number']==2
    runner.bundle_session(status)
    output=tmp_path/'BT_UNET_SESSION_ALL_0001_TO_0001.zip'
    assert output.is_file()
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert 'completed_jobs/'+archive.name in z.namelist()
        manifest=json.loads(z.read('session_bundle_manifest.json'))
        assert manifest['last_completed_experiment_number']==1
        assert manifest['next_experiment_number']==2


def test_failed_experiment_does_not_advance_counter(tmp_path,monkeypatch):
    runner=load_runner(tmp_path,monkeypatch,start=8)
    status={'scope':'ALL','next_index':7,'completed_ids':[],
            'session_archives':[],'stop_reason':'blocked_or_failed'}
    runner._save_queue(status)
    assert (tmp_path/'BT_UNET_NEXT_EXPERIMENT_ALL.txt').read_text().strip()=='8'


def test_merger_detects_missing_intervals(tmp_path,monkeypatch):
    runner=load_runner(tmp_path,monkeypatch,start=2)
    runner.export_plan()
    runner.SESSION.mkdir(parents=True,exist_ok=True)
    archive=tmp_path/'BT_UNET_RESULTS__test_job.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('x.txt','ok')
    status={'scope':'ALL','next_index':2,'completed_ids':['job2'],
            'session_archives':[str(archive)],'git_commit':'a'*40,
            'stop_reason':'test'}
    runner._save_queue(status);runner.bundle_session(status)
    from scripts.kaggle.merge_sessions import collect
    try:
        collect(tmp_path)
    except ValueError as exc:
        assert 'expected 1' in str(exc)
    else:
        raise AssertionError('Missing first experiment was accepted')


def _write_fake_kvasir_zip(path):
    """Layout-only fixture: 1000 pairs, no real medical images or network access."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        for n in range(1000):
            z.writestr(f'Kvasir-SEG/images/{n:04d}.jpg',b'fake image')
            z.writestr(f'Kvasir-SEG/masks/{n:04d}.jpg',b'fake mask')


def test_kvasir_mount_preferred_no_network(tmp_path,monkeypatch):
    runner=load_runner(tmp_path,monkeypatch)
    mount=tmp_path/'input'/'kvasirseg'
    archive=mount/'kvasir-seg.zip'
    _write_fake_kvasir_zip(archive)
    monkeypatch.setenv('BT_KAGGLE_INPUT_ROOT',str(tmp_path/'input'))
    monkeypatch.setattr(runner,'run_cmd',lambda *a,**kw: (_ for _ in ()).throw(
        AssertionError('unexpected network download')))
    resolved,provenance=runner._resolve_kvasir_kaggle()
    assert resolved[0]=='--zip-path'
    assert Path(resolved[1])==archive
    assert provenance=='kaggle_input_mount'


def test_kvasir_official_direct_avoids_kagglehub(tmp_path,monkeypatch):
    runner=load_runner(tmp_path,monkeypatch)
    monkeypatch.setenv('BT_KAGGLE_INPUT_ROOT',str(tmp_path/'missing_input'))
    def synthetic_direct(cmd,**kwargs):
        assert kwargs['name']=='download_kvasir_official'
        assert 'datasets.simula.no/downloads/kvasir-seg.zip' in str(cmd)
        assert 'kagglehub' not in str(cmd)
        dst=runner.SESSION_DATA/'downloads'/'kvasir_seg'/'kvasir-seg.zip'
        _write_fake_kvasir_zip(dst)
    monkeypatch.setattr(runner,'run_cmd',synthetic_direct)
    result,provenance=runner._resolve_kvasir_kaggle()
    assert result[0]=='--zip-path'
    assert result[2] and len(result[2])==64
    assert provenance=='official_simula_https'


def test_failed_dataset_download_does_not_advance(tmp_path,monkeypatch):
    runner=load_runner(tmp_path,monkeypatch)
    monkeypatch.setenv('BT_KAGGLE_INPUT_ROOT',str(tmp_path/'missing_input'))
    def fail(*a,**kw):
        raise RuntimeError('simulated network failure')
    monkeypatch.setattr(runner,'run_cmd',fail)
    import pytest
    with pytest.raises(RuntimeError,match='Add Input'):
        runner._resolve_kvasir_kaggle()
    assert 'kagglehub' not in runner._resolve_kvasir_kaggle.__code__.co_names
