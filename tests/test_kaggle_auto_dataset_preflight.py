"""Offline tests for Kaggle ZIP probes and honest SKIPPED job bookkeeping."""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PREFLIGHT=ROOT/'scripts/kaggle/dataset_preflight.py'
RUNNER=ROOT/'scripts/kaggle/multi_t4_runner.py'


def _module(filename,name):
    spec=importlib.util.spec_from_file_location(name,filename)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runner(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['multi_t4_runner.py','--scope','AUTO',
                                    '--one-based','--start-index','1','--bundle-session'])
    monkeypatch.setenv('BT_KAGGLE_WORKDIR',str(tmp_path))
    return _module(RUNNER,'test_auto_preflight_runner')


def test_preflight_only_official_direct_urls():
    mod=_module(PREFLIGHT,'test_dataset_source_urls')
    for ds in mod.AUTO_DATASETS:
        urls=mod.candidate_urls(ds)
        assert (urls or ds == "bus_uclm") and all(u.startswith('https://') for u in urls)
        assert not any('drive.google.com' in u or 'kaggle.com' in u for u in urls)
    assert len(mod.candidate_urls('hyper_kvasir_seg'))==2
    assert not mod.candidate_urls('bus_uclm')  # uses HF/Kaggle SDK, not ZIP probe


def test_probe_rejects_html_without_full_download(monkeypatch):
    mod=_module(PREFLIGHT,'test_probe_rejects_html')
    class FakeResponse:
        status_code=200
        url='https://downloads.imagej.net/foo.zip'
        headers={}
        def raise_for_status(self):pass
        def iter_content(self,chunk_size=4):
            assert chunk_size==4
            yield b'<htm'
        def __enter__(self):return self
        def __exit__(self,*args):pass
    import requests
    monkeypatch.setattr(requests,'get',lambda *a,**k:FakeResponse())
    try:
        mod.probe_url('https://downloads.imagej.net/foo.zip')
    except RuntimeError as exc:
        assert 'ZIP' in str(exc)
    else: raise AssertionError('HTML must not pass ZIP probe')


def test_probe_zip_magic_requires_only_four_bytes(monkeypatch):
    mod=_module(PREFLIGHT,'test_probe_valid_zip')
    class FakeResponse:
        status_code=206
        url='https://downloads.imagej.net/isbi.zip'
        headers={'Content-Length':'4'}
        def raise_for_status(self):pass
        def iter_content(self,chunk_size=4):
            assert chunk_size==4
            yield b'PK\x03\x04'
        def __enter__(self):return self
        def __exit__(self,*args):pass
    import requests
    monkeypatch.setattr(requests,'get',lambda *a,**k:FakeResponse())
    info=mod.probe_url('https://downloads.imagej.net/isbi.zip')
    assert info['range_supported'] is True
    assert info['https_certificate_verified'] is True


def test_source_fallback_does_not_claim_dataset_download():
    mod=_module(PREFLIGHT,'test_preflight_fallback')
    def fake_probe(url):
        if 'datasets.simula.no' in url:
            raise RuntimeError('SSL failure')
        return {'url':url,'https_certificate_verified':True}
    r=mod.check_sources(['hyper_kvasir_seg'],probe=fake_probe)
    selected=r['hyper_kvasir_seg']
    assert selected['available']
    assert selected['selected']['url'].startswith('https://huggingface.co/')
    assert selected['errors']


def test_unavailable_dataset_is_explicitly_skipped_not_completed(tmp_path,monkeypatch):
    r=_runner(tmp_path,monkeypatch)
    r.export_plan()
    r.SESSION.mkdir(parents=True,exist_ok=True)
    status={'scope':'AUTO','next_index':0,'completed_ids':[],'completed_numbers':[],
            'skipped_jobs':[],'session_archives':[],
            'git_commit':'b'*40,'stop_reason':None}
    r.AUTO_UNAVAILABLE['kvasir_seg']='HTTP 403 during preflight'
    r._auto_skip(status,0,r.QUEUE[0])
    assert status['next_index']==1
    assert status['completed_ids']==[]
    assert len(status['skipped_jobs'])==1
    assert (tmp_path/'BT_UNET_NEXT_EXPERIMENT_AUTO.txt').read_text().strip()=='2'
    r.bundle_session(status)
    archive=tmp_path/'BT_UNET_SESSION_AUTO_0001_TO_0001.zip'
    assert archive.is_file()
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('session_bundle_manifest.json'))
        assert manifest['last_completed_experiment_number']==0
        assert manifest['last_processed_experiment_number']==1
        assert manifest['skipped_jobs'][0]['number']==1
        assert not manifest['completed_ids']
    from scripts.kaggle.merge_sessions import collect, merge
    packages,nxt=collect(tmp_path,scope='AUTO')
    assert len(packages)==1 and nxt==2
    try:
        merge(tmp_path,tmp_path/'should_fail.zip',require_complete=True,scope='AUTO')
    except ValueError as exc:
        assert 'skipped' in str(exc)
    else:
        raise AssertionError('A skipped job must never satisfy require-complete')


def test_new_bus_uclm_jobs_are_independent(tmp_path,monkeypatch):
    r=_runner(tmp_path,monkeypatch)
    bus=[spec for spec in r.QUEUE if spec['dataset']=='bus_uclm']
    assert len(bus)==12
    assert all(job['id'].startswith('standard__bus_uclm__') for job in bus)
    assert len({job['id'] for job in r.QUEUE})==165
