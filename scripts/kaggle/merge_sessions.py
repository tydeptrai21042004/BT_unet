#!/usr/bin/env python3
"""Merge finished Kaggle session ZIPs without changing results or dropping failures.

Never identifies a failed or missing job as complete. Can be used on a local computer
with downloaded session archives. Example:
  python scripts/kaggle/merge_sessions.py --input-dir results_downloads \
      --output BT_UNET_ALL_261_RESULTS.zip --require-complete
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha256_stream(reader):
    h=hashlib.sha256()
    for chunk in iter(lambda:reader.read(1024*1024),b''):
        h.update(chunk)
    return h.hexdigest()


def collect(input_dir:Path, scope='ALL'):
    packages=[]
    for file in sorted(input_dir.glob(f'BT_UNET_SESSION_{scope}_*.zip')):
        with zipfile.ZipFile(file) as z:
            m=json.loads(z.read('session_bundle_manifest.json'))
            archives=m['archives']
            if not archives and not m.get('completed_ids') and not m.get('skipped_jobs'):
                print('[SKIP EMPTY SESSION]',file.name,'no completed or skipped experiments')
                continue
            if len(archives)!=len(m['completed_ids']):
                raise ValueError(f'Archive count mismatch: {file}')
            for a in archives:
                inner='completed_jobs/'+a['name']
                if a.get('format') == 'compact_metrics':
                    paper_row=inner+'/paper_row.json'
                    if paper_row not in z.namelist():
                        raise ValueError(f'Missing compact result {paper_row}: {file}')
                    row=json.loads(z.read(paper_row))
                    if int(row['experiment_number']) not in m.get('completed_experiment_numbers',[]):
                        raise ValueError(f'Unlisted completed experiment {paper_row}: {file}')
                else:
                    if inner not in z.namelist():
                        raise ValueError(f'Missing {inner}: {file}')
                    with z.open(inner) as f:
                        actual=sha256_stream(f)
                    if actual!=a['sha256']:
                        raise ValueError(f'Bad checksum for {inner} in {file}')
            packages.append((file,m))
    if not packages:
        raise FileNotFoundError(f'No BT_UNET_SESSION_{scope}_*.zip found in {input_dir}')
    packages.sort(key=lambda item:item[1]['first_experiment_number'])
    if any(m.get('scope') != scope for _,m in packages):
        raise ValueError('Session scope mismatch')
    hashes={m['plan_sha256'] for _,m in packages}
    commits={m['git_commit'] for _,m in packages}
    if len(hashes)!=1 or len(commits)!=1:
        raise ValueError('Sessions use different experiment plans/GitHub commits; do not merge')
    next_index=1
    for file,m in packages:
        if m['first_experiment_number']!=next_index:
            raise ValueError(f'Missing/overlapping experiment numbers: expected {next_index}, '
                             f'got {m["first_experiment_number"]} in {file}')
        last=m.get('last_processed_experiment_number',m['last_completed_experiment_number'])
        if last < next_index-1:raise ValueError('Invalid last-processed value')
        skipped = m.get('skipped_jobs',[])
        completed_numbers=m.get('completed_experiment_numbers',list(range(next_index,last+1)))
        completed_set=set(completed_numbers)
        skipped_set={item['number'] for item in skipped}
        if completed_set.intersection(skipped_set):
            raise ValueError(f'Experiment cannot be both skipped and completed: {file}')
        if completed_set.union(skipped_set) != set(range(next_index,last+1)):
            raise ValueError(f'Missing or duplicated processed experiment numbers: {file}')
        if len(completed_set)!=len(m['completed_ids']):
            raise ValueError(f'Completed-ID count mismatch: {file}')
        next_index=last+1
    return packages,next_index


def merge(input_dir,output,require_complete=False,scope='ALL'):
    input_dir=Path(input_dir);output=Path(output)
    packages,next_exp=collect(input_dir,scope=scope)
    total=packages[0][1].get('total_experiments',261 if scope=='ALL' else None)
    if total is None or any(m.get('total_experiments',total)!=total for _,m in packages):
        raise ValueError('Unknown or inconsistent queue length')
    total_completed=sum(len(m['completed_ids']) for _,m in packages)
    total_skipped=sum(len(m.get('skipped_jobs',[])) for _,m in packages)
    if require_complete and (next_exp!=total+1 or total_skipped):
        raise ValueError(f'Expected {total} completed experiments; '
                         f'{total_completed} completed, {total_skipped} skipped; '
                         f'next experiment is {next_exp}')
    if output.exists():raise FileExistsError(output)
    if output.parent.resolve()==input_dir.resolve() and output.name.startswith(f'BT_UNET_SESSION_{scope}_'):
        raise ValueError('Choose an aggregate name different from a session package')
    needed=sum(Path(file).stat().st_size for file,_ in packages)
    if shutil.disk_usage(output.parent).free<needed+100*1024*1024:
        raise OSError('Not enough free disk space for another copy of the session ZIP contents')
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix(output.suffix+'.partial')
    completed=[]
    try:
        with zipfile.ZipFile(partial,'w',allowZip64=True) as target:
            overview={'source_git_sha':packages[0][1]['git_commit'],
                      'plan_sha256':packages[0][1]['plan_sha256'],
                      'completed_experiments':total_completed,
                      'skipped_experiments':total_skipped,
                      'processed_experiments':next_exp-1,
                      'next_experiment':next_exp,
                      'session_files':[f.name for f,_ in packages],
                      'total_experiments':total,
                      'includes_complete_queue':next_exp==total+1 and total_skipped==0}
            target.writestr('MERGE_MANIFEST.json',json.dumps(overview,indent=2))
            for file,manifest in packages:
                with zipfile.ZipFile(file) as session:
                    target.writestr('session_metadata/'+file.stem+'.json',
                                    json.dumps(manifest,indent=2))
                    for archive in manifest['archives']:
                        entry='completed_jobs/'+archive['name']
                        if archive.get('format') == 'compact_metrics':
                            # Compact sessions store text files directly, no nested
                            # per-job ZIP or large checkpoint in the merged archive.
                            entries=[name for name in session.namelist() if name.startswith(entry+'/')]
                            if not entries or (entry+'/paper_row.json') not in entries:
                                raise ValueError('Missing compact experiment '+entry)
                            for name in entries:
                                with session.open(name) as src, target.open(name,'w',force_zip64=True) as dst:
                                    shutil.copyfileobj(src,dst,1024*1024)
                        else:
                            with session.open(entry) as src, target.open(entry,'w',force_zip64=True) as dst:
                                shutil.copyfileobj(src,dst,1024*1024)
                        completed.append(archive['name'])
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:raise RuntimeError('Merged archive CRC check failed')
            found=[x for x in z.namelist()
                   if x.startswith('completed_jobs/') and (x.endswith('/paper_row.json') or x.endswith('.zip'))]
            if len(found)!=len(completed):
                raise RuntimeError('Merged archive missing completed jobs')
        partial.replace(output)
    finally:
        if partial.exists(): partial.unlink()
    print('Created:',output)
    print('Successfully completed:',total_completed,f'/{total}; skipped:',total_skipped,
          '; next experiment:',next_exp)
    print('Git SHA:',packages[0][1]['git_commit'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--require-complete',action='store_true')
    ap.add_argument('--scope',choices=['ALL','AUTO'],default='ALL')
    args=ap.parse_args()
    merge(args.input_dir,args.output,args.require_complete,args.scope)

if __name__=='__main__':main()
