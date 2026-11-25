"""Regression tests: real test loss, complete-only paper rows, compact bundles."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


def _summary(spec):
    return {
        'checkpoint_sha256': 'a'*64, 'model': spec['model'],
        'dataset': spec['dataset'], 'training_seed': int(spec['seed']),
        'split': 'test', 'corruption': 'clean', 'removed_frequency_band': 'none',
        'num_samples': 3, 'metrics_mean': {
            'dice': .81, 'iou': .75, 'precision': .90, 'recall': .79,
            'mae': .10, 'loss': .20, 'boundary_f1': .5,
            'hd95_pixels': 2., 'assd_pixels': 1., 'pixel_accuracy': .94,
            'specificity': .91, 'balanced_accuracy': .85, 'f2_score': .80,
        }, 'test_loss': .20,
    }


def test_export_and_complete_only_csv(tmp_path):
    from scripts.kaggle.results_export import export_completed, update_reports
    spec = {'id': 'one', 'dataset': 'kvasir_seg', 'model': 'unet', 'variant': 'convergence_unet',
            'seed': 42, 'epochs': 100}
    out = tmp_path / 'test_eval'
    out.mkdir()
    (out / 'summary.json').write_text(json.dumps(_summary(spec)))
    (out / 'per_image.csv').write_text('sample_id,dice\na,0.8\n')
    evals = [('kvasir_seg', ['python', 'eval.py', '--output-dir', str(out)])]
    record, folder = export_completed(1, spec, evals, tmp_path/'jobs', tmp_path/'compact', {'status': 'complete'})
    assert record['loss'] == '0.200000'
    plan = [spec, {**spec, 'id': 'two', 'seed': 1337}]
    completed = update_reports(tmp_path, tmp_path/'compact', plan)
    assert len(completed) == 1
    with (tmp_path/'BT_UNET_PAPER_RESULTS_AUTO.csv').open() as fp:
        rows = list(csv.DictReader(fp))
    assert rows[0]['loss'] == '0.200000' and rows[0]['dice'] == '0.810000'
    assert rows[1]['loss'] == '---' and rows[1]['status'] == 'pending'
    assert (folder/'evaluations/kvasir_seg/per_image.csv').is_file()
    assert not list(folder.rglob('*.pt'))


def test_all_revision_auto_targets_use_real_dataset_labels(tmp_path, monkeypatch):
    """Regression: evaluator output dataset must match its export label.

    Previously make_job() labeled revision output as 'test' even though
    summary.json correctly identified dataset='kvasir_seg'. After a completed
    100-epoch run, the exporter rejected the mismatch and lost the session.
    """
    runner_script = ROOT / 'scripts/kaggle/multi_t4_runner.py'
    monkeypatch.setattr(sys, 'argv', ['multi_t4_runner.py', '--scope', 'AUTO',
                                    '--one-based', '--start-index', '1', '--plan-only'])
    monkeypatch.setenv('BT_KAGGLE_WORKDIR', str(tmp_path))
    module_spec = importlib.util.spec_from_file_location('revision_target_runner', runner_script)
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    revisions = [job for job in runner.QUEUE if job['type'] == 'revision']
    assert len(revisions) == 57
    for job in revisions:
        runner._activate(job)
        _, _, evaluations, _, _ = runner.make_job()
        assert len(evaluations) == 1
        label, command = evaluations[0]
        assert label == job['dataset'], job['id']
        assert command[command.index('--dataset') + 1] == label
        assert command[command.index('--split') + 1] == 'test'


def test_incomplete_loss_is_not_claimed_complete(tmp_path):
    from scripts.kaggle.results_export import export_completed
    spec={'id':'one','dataset':'kvasir_seg','model':'unet','seed':42,'epochs':100}
    data=_summary(spec)
    data['metrics_mean'].pop('loss')
    data.pop('test_loss')
    out=tmp_path/'eval'
    out.mkdir()
    (out/'summary.json').write_text(json.dumps(data))
    (out/'per_image.csv').write_text('sample_id,dice\na,0.8\n')
    with pytest.raises(ValueError,match='six-paper-metric'):
        export_completed(1,spec,[('kvasir_seg',['--output-dir',str(out)])],tmp_path/'jobs',tmp_path/'compact',{})


def test_evaluator_calculates_configured_test_loss(tmp_path,monkeypatch):
    revision = ROOT/'scripts/revision'
    sys.path.insert(0,str(revision))
    import evaluate_revision as er
    class FakeModel(torch.nn.Module):
        def forward(self,images):
            return torch.zeros((len(images),1,16,16),device=images.device)
    class FakeDataset(Dataset):
        def __len__(self): return 3
        def __getitem__(self,index):
            return {'image': torch.zeros(3,16,16), 'mask': torch.ones(1,16,16), 'id': f'm{index}'}
    monkeypatch.setattr(er,'load_checkpoint', lambda *a: (FakeModel(),
        {'train':{'seed':42,'loss':'bce_dice'},'data':{'image_size':16}}, 'unet'))
    monkeypatch.setattr(er,'manifest_path',lambda *a: None)
    monkeypatch.setattr(er,'build_dataset',lambda **kw: FakeDataset())
    monkeypatch.setattr(er,'build_eval_transforms',lambda **kw: None)
    monkeypatch.setattr(er,'CorruptThenTransform',lambda *a: None)
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'fake weights only for hash')
    args=Namespace(checkpoint=str(checkpoint),model=None,config=None,device='cpu',
        image_size=16,corruption='clean',severity=1,seed=42,data_root=str(tmp_path),
        dataset='kvasir_seg',split='test',batch_size=2,num_workers=0,
        cluster_map=None,remove_frequency_band=None,output_dir=str(tmp_path/'out'),
        threshold=.5,boundary_tolerance=2,save_predictions=False)
    er.run(args)
    data=json.loads((tmp_path/'out/summary.json').read_text())
    from scripts.train_one import build_loss
    expect=build_loss('bce_dice')(torch.zeros(2,1,16,16),torch.ones(2,1,16,16)).item()
    assert data['test_loss'] == pytest.approx(expect,rel=1e-5)
    assert data['metrics_mean']['loss'] == pytest.approx(expect,rel=1e-5)
    assert data['metrics_mean']['pixel_accuracy'] == 1.
    assert data['metrics_mean']['specificity'] == 0.
    assert data['test_loss_num_batches'] == 2


def test_compact_auto_session_zip(tmp_path,monkeypatch):
    from scripts.kaggle.results_export import export_completed, update_reports
    runner_script=ROOT/'scripts/kaggle/multi_t4_runner.py'
    monkeypatch.setattr(sys,'argv',['multi_t4_runner.py','--scope','AUTO','--one-based',
                                  '--start-index','1','--bundle-session'])
    monkeypatch.setenv('BT_KAGGLE_WORKDIR',str(tmp_path))
    spec=importlib.util.spec_from_file_location('compactrunner',runner_script)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    runner.export_plan()
    base_spec=runner.QUEUE[0]
    out=tmp_path/'eval';out.mkdir()
    (out/'summary.json').write_text(json.dumps(_summary(base_spec)))
    (out/'per_image.csv').write_text('sample_id,dice\na,0.8\n')
    record,dir_path=export_completed(1,base_spec,[('kvasir_seg',['--output-dir',str(out)])],
                                     tmp_path/'jobs',runner.COMPACT_RESULTS,{'git_commit':'f'*40})
    update_reports(tmp_path,runner.COMPACT_RESULTS,runner.QUEUE)
    status={'scope':'AUTO','next_index':1,'completed_ids':[base_spec['id']],
            'completed_numbers':[1], 'session_archives':[str(dir_path)],
            'git_commit':'f'*40, 'stop_reason':'budget'}
    runner._save_queue(status)
    runner.bundle_session(status)
    archive=tmp_path/'BT_UNET_SESSION_AUTO_0001_TO_0001.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert 'BT_UNET_PAPER_RESULTS_AUTO.csv' in names
        assert 'BT_UNET_COMPLETED_RESULTS_AUTO.csv' in names
        assert any(x.endswith('paper_row.json') for x in names)
        assert any(x.endswith('per_image.csv') for x in names)
        assert not any(x.endswith('.pt') or x.endswith('.png') for x in names)
        assert not any(x.endswith('.zip') for x in names)
    assert archive.stat().st_size < 400_000
    from scripts.kaggle.merge_sessions import collect, merge
    packs,next_exp=collect(tmp_path,scope='AUTO')
    assert len(packs)==1 and next_exp==2
    merged=tmp_path/'merged.zip'
    merge(tmp_path,merged,scope='AUTO')
    with zipfile.ZipFile(merged) as z:
        assert any(s.endswith('paper_row.json') for s in z.namelist())
    from scripts.kaggle.merge_paper_metrics import merge_paper_results
    all_rows=merge_paper_results(tmp_path,tmp_path/'filled_paper')
    assert len(all_rows)==165 and all_rows[0]['loss']=='0.200000'
    assert all_rows[1]['loss']=='---'


def test_auto_budget_allows_epoch_recovery_instead_of_permanent_deferral(tmp_path,monkeypatch):
    runner_script=ROOT/'scripts/kaggle/multi_t4_runner.py'
    monkeypatch.setattr(sys,'argv',['multi_t4_runner.py','--scope','AUTO','--one-based',
                                   '--start-index','2','--bundle-session'])
    monkeypatch.setenv('BT_KAGGLE_WORKDIR',str(tmp_path))
    spec=importlib.util.spec_from_file_location('estimate_runner',runner_script)
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    base={'experiment':{'name':'unit'}, 'model':{'name':'unet'},
          'train':{'epochs':100,'seed':1337},'data':{'dataset':'kvasir_seg'}}
    key=r._calibration_key(base)
    r.CALIBRATION_CACHE[key]=130.0
    r._activate(r.QUEUE[1])
    monkeypatch.setattr(r,'remaining',lambda:7.56*3600)
    assert r._preflight([],base,None,[('test',[])]) is None
    assert r.RUN_META['time_policy']=='epoch_recovery_no_cold_start_calibration'
    r.CALIBRATION_CACHE[key]=121.3
    monkeypatch.setattr(r,'remaining',lambda:4.1*3600)
    assert r._preflight([],base,None,[('test',[])]) is None
    assert r.args.max_hours==12.0
