"""Import already completed compact AUTO rows from explicit archived evidence.
Archive identity checks do not prove that old sessions used identical sample membership.
Original archives remain the source of evidence; no old scores are synthesized.
"""
from __future__ import annotations
import csv, io, json, math, zipfile
from pathlib import Path
from scripts.kaggle.results_export import MAIN

def compatible_plan(source, current):
    if len(source) in (165,300,324) and source == current[:len(source)]:
        return True
    if len(current)==324 and len(source) in (165,300):
        # Compare against the shipped, immutable pre-revision plan, never accept
        # old results just because an experiment number happens to match.
        legacy_path=Path(__file__).resolve().parents[2]/'BT_UNET_PLAN_AUTO_300.csv'
        with legacy_path.open(newline='') as f: legacy=list(csv.DictReader(f))
        return source==legacy[:len(source)]
    return False

def read_archives(directory, current_plan, current_specs=None):
    records={}
    by_id={r['id']: (i,r) for i,r in enumerate(current_plan,1)}
    paths=sorted(Path(directory).rglob('BT_UNET_SESSION_AUTO_*.zip'))
    for archive in paths:
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('session_bundle_manifest.json'))
            if manifest.get('scope')!='AUTO': continue
            plan=list(csv.DictReader(io.StringIO(z.read('BT_UNET_PLAN_AUTO.csv').decode())))
            if not compatible_plan(plan,current_plan):
                raise ValueError(f'Incompatible plan in {archive}')
            listed=set(map(int,manifest.get('completed_experiment_numbers',[])))
            found=set()
            for entry in manifest.get('archives',[]):
                if entry.get('format')!='compact_metrics':
                    continue # original nested archives need the separate merge tool
                prefix='completed_jobs/'+entry['name']+'/'
                row=json.loads(z.read(prefix+'paper_row.json'))
                n=int(row['experiment_number'])
                if n not in listed or n in found or not 1<=n<=len(plan): raise ValueError(f'Invalid index in {archive}: {n}')
                expected=plan[n-1]
                for k in ('dataset','model','variant','seed','epochs'):
                    if str(row.get(k,''))!=str(expected[k]): raise ValueError(f'Row {n}: {k} mismatch')
                if row['experiment_id']!=expected['id'] or row['status']!='complete' or any(not math.isfinite(float(row[k])) for k in MAIN): raise ValueError(f'Invalid complete row {n}')
                evidence=json.loads(z.read(prefix+'provenance.json'))
                if evidence.get('experiment',{}).get('id')!=expected['id'] or not evidence.get('checkpoint_sha256') or evidence.get('test_split')!='test': raise ValueError(f'Invalid provenance for {n}')
                found.add(n)
                mapped=by_id.get(expected['id'])
                if mapped is None:
                    continue  # old proposal or retired comparison, never import into revised queue
                target_n,target=mapped
                if expected!=target: raise ValueError(f'Current plan metadata mismatch for {n}')
                if current_specs is not None and evidence['experiment']!=current_specs[target_n-1]: raise ValueError(f'Full experiment configuration mismatch for {n}')
                if expected['id'].startswith('h2__') and evidence['experiment'].get('fourier_hermitian_constraint') is not True:
                    raise ValueError(f'Missing revised operator provenance for {n}')
                # Recover failure rates and existing per-image metrics without running the model.
                per_image=prefix+'evaluations/'+expected['dataset']+'/per_image.csv'
                if per_image in z.namelist():
                    per=list(csv.DictReader(io.StringIO(z.read(per_image).decode())))
                    if per and len(per)==evidence.get('test_sample_count'):
                        if {'complete_miss','empty_ground_truth','empty_prediction','distance_defined'}.issubset(per[0]):
                            positives=sum(int(float(r['empty_ground_truth']))==0 for r in per)
                            if positives and row.get('complete_miss_rate','---')=='---': row['complete_miss_rate']=f"{sum(int(float(r['complete_miss'])) for r in per)/positives:.6f}"
                            if row.get('empty_prediction_rate','---')=='---': row['empty_prediction_rate']=f"{sum(int(float(r['empty_prediction'])) for r in per)/len(per):.6f}"
                            if row.get('undefined_distance_rate','---')=='---': row['undefined_distance_rate']=f"{sum(int(float(r['distance_defined']))==0 for r in per)/len(per):.6f}"
                if n in records and any(str(records[n][0].get(k))!=str(row.get(k)) for k in MAIN): raise ValueError(f'Conflicting completed row {n}; resolve archives explicitly')
                records[n]=(row, {'source_archive':str(archive),'source_provenance':evidence,'source_commit':manifest.get('git_commit'),'reuse_note':'Identity verified; sample membership across sessions must still be audited.'})
                found.add(n)
            if any(e.get('format')=='compact_metrics' for e in manifest.get('archives',[])) and found!=listed: raise ValueError(f'Incomplete compact archive {archive}')
    return records

def import_rows(directory, plan_csv, compact_root, current_specs=None):
    with Path(plan_csv).open(newline='') as f: plan=list(csv.DictReader(f))
    rows=read_archives(directory,plan,current_specs)
    for n,(row,provenance) in rows.items():
        dest=Path(compact_root)/f'{n:04d}'; dest.mkdir(parents=True,exist_ok=True)
        existing=dest/'paper_row.json'
        if existing.exists() and json.loads(existing.read_text())!=row: raise ValueError(f'Conflicting local result {n}')
        existing.write_text(json.dumps(row,indent=2)+'\n')
        (dest/'reuse_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return set(rows)
