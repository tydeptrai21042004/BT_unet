#!/usr/bin/env python3
"""Merge compact and legacy Kaggle AUTO session ZIPs into a 165-run results CSV.

Example:
  python scripts/kaggle/merge_paper_metrics.py --input-dir downloaded_sessions \\
      --output-dir paper_results --allow-mixed-commits

Legacy (pre-patch) evaluations cannot provide test loss; it is left as ---.
Check each session's plan hash and every experiment ID before accepting results.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kaggle.results_export import COLUMNS, MAIN, EXTRA, _metric, _write_csv


def merge_paper_results(input_dir, output_dir, *, allow_mixed_commits=False):
    input_dir, output_dir=Path(input_dir),Path(output_dir)
    archives=sorted(input_dir.glob('BT_UNET_SESSION_AUTO_*.zip'))
    if not archives:
        raise FileNotFoundError('No BT_UNET_SESSION_AUTO_*.zip archives found')
    plan=None
    plan_hash=None
    commits=set()
    data={}
    evidence={}
    for archive in archives:
        with zipfile.ZipFile(archive) as session:
            manifest=json.loads(session.read('session_bundle_manifest.json'))
            if manifest.get('scope')!='AUTO':
                raise ValueError('Not an AUTO session: '+archive.name)
            if plan_hash is None:
                plan_hash=manifest['plan_sha256']
            if manifest['plan_sha256'] != plan_hash:
                raise ValueError('Different 165-run experiment plans cannot be merged')
            commits.add(manifest.get('git_commit'))
            with session.open('BT_UNET_PLAN_AUTO.csv') as handle:
                records=list(csv.DictReader(io.TextIOWrapper(handle,encoding='utf-8')))
            if len(records)!=165:
                raise ValueError(f'Expected 165 plan rows in {archive.name}')
            if plan is None:
                plan=records
            elif plan != records:
                raise ValueError(f'Experiment ordering or metadata changed in {archive.name}')
            listed=set(int(x) for x in manifest.get('completed_experiment_numbers',[]))
            found=set()
            for item in manifest.get('archives',[]):
                if item.get('format')=='compact_metrics':
                    name='completed_jobs/'+item['name']+'/paper_row.json'
                    if name not in session.namelist():
                        raise ValueError(f'Missing compact paper row: {name}')
                    row=json.loads(session.read(name))
                    n=int(row['experiment_number'])
                else:
                    inner_name='completed_jobs/'+item['name']
                    with zipfile.ZipFile(io.BytesIO(session.read(inner_name))) as inner:
                        test_files=[f for f in inner.namelist()
                                    if f.endswith('/metrics/test/summary.json') or
                                       (f.endswith('/summary.json') and '/evaluations/' in f)]
                        if len(test_files)!=1:
                            raise ValueError(f'Expected one primary clean test evaluation: {archive.name}: {test_files}')
                        summary=json.loads(inner.read(test_files[0]))
                    n=next((i for i,r in enumerate(plan,1) if r['id']==item['name'].removeprefix('BT_UNET_RESULTS__').removesuffix('.zip')),None)
                    if n is None:
                        raise ValueError(f'Cannot map legacy experiment: {item["name"]}')
                    row={'experiment_number':n, 'experiment_id':plan[n-1]['id'],
                         'dataset':summary.get('dataset'), 'model':plan[n-1]['model'],
                         'variant':plan[n-1]['variant'], 'seed':summary.get('training_seed'),
                         'epochs':plan[n-1]['epochs'],
                         'status':'complete_legacy_missing_test_loss' if 'loss' not in summary.get('metrics_mean',{}) else 'complete'}
                    row.update({name:_metric(summary,name) for name in MAIN+EXTRA})
                    if summary.get('split')!='test' or summary.get('corruption')!='clean':
                        raise ValueError('Legacy summary is not a clean test evaluation')
                if n not in listed or n in found or n in data:
                    raise ValueError(f'Unexpected or duplicate completed run #{n} from {archive.name}')
                expected=plan[n-1]
                if str(row['experiment_id'])!=expected['id'] or str(row['dataset'])!=expected['dataset'] or int(row['seed'])!=int(expected['seed']) or int(row['epochs'])!=int(expected['epochs']):
                    raise ValueError('Experiment identity mismatch in '+archive.name)
                data[n]=row
                evidence[n]=archive.name
                found.add(n)
            if found != listed:
                raise ValueError(f'Session completed count mismatch in {archive.name}: {sorted(found)} != {sorted(listed)}')
    if len(commits)>1 and not allow_mixed_commits:
        raise ValueError('Different Git commits. If code changes only reporting, explicitly pass --allow-mixed-commits after review')
    completed=[];rows=[]
    for i,entry in enumerate(plan,1):
        row={'experiment_number':i,'experiment_id':entry['id'],
             'dataset':entry['dataset'], 'model':entry['model'],
             'variant':entry['variant'],'seed':entry['seed'], 'epochs':entry['epochs'],
             'status':'pending'}
        row.update({name:'---' for name in MAIN+EXTRA})
        if i in data:
            row.update(data[i]);completed.append(row)
        rows.append(row)
    output_dir.mkdir(parents=True,exist_ok=True)
    _write_csv(output_dir/'BT_UNET_PAPER_RESULTS_AUTO.csv',rows)
    _write_csv(output_dir/'BT_UNET_COMPLETED_RESULTS_AUTO.csv',completed)
    (output_dir/'BT_UNET_COMPLETED_RESULTS_AUTO.json').write_text(json.dumps(completed,indent=2)+'\n')
    tex=[]
    for row in completed:
        vals=' & '.join('\\('+row[name]+'\\)' if row[name]!='---' else r'\texttt{---}' for name in MAIN)
        tex.append('% Experiment #{:d}: {}'.format(int(row['experiment_number']), row['experiment_id']))
        tex.append('% Replace its 6 metric cells with: '+vals)
    (output_dir/'BT_UNET_PAPER_VALUES_AUTO.tex').write_text('\n'.join(tex)+'\n')
    (output_dir/'MERGE_PROVENANCE_AUTO.json').write_text(json.dumps({
        'plan_sha256':plan_hash, 'commits':sorted(str(v) for v in commits),
        'source_archives':[x.name for x in archives], 'experiment_sources':evidence,
        'completed_count':len(completed), 'missing_count':165-len(completed),
        'missing_test_loss_count':sum(r['loss']=='---' for r in completed),
    },indent=2)+'\n')
    print('[MERGE] completed',len(completed),'of 165; missing test loss:',
          sum(r['loss']=='---' for r in completed))
    print('[MERGE OUTPUT]',output_dir/'BT_UNET_PAPER_RESULTS_AUTO.csv')
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--allow-mixed-commits',action='store_true')
    args=ap.parse_args()
    merge_paper_results(args.input_dir,args.output_dir,allow_mixed_commits=args.allow_mixed_commits)


if __name__=='__main__':
    main()
