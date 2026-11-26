"""Acquire the two original datasets without manual Kaggle input or credentials on HF.
All candidates must pass the full expected pair count before use.
"""
from __future__ import annotations
import argparse, json, re, shutil, sys, tempfile, zipfile
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.prepare_kvasir_seg import _collect_pairs, _find_dataset_root, _maybe_download
from src.datasets.kvasir_seg_dataset import _resolve_image_mask_dirs, canonical_sample_id
from src.datasets import load_binary_mask
from PIL import Image

EXPECTED = {'kvasir_instrument': 590, 'bus_uclm': 683, 'cvc_clinicdb':612, 'cvc_colondb':380, 'etis':196, 'isic2018':2594}
KAGGLE = {'kvasir_instrument':'debeshjha1/kvasirinstrument', 'bus_uclm':'orvile/bus-uclm-breast-ultrasound-dataset', 'cvc_clinicdb':'balraj98/cvcclinicdb', 'cvc_colondb':'longvil/cvc-colondb', 'etis':'nguyenvoquocduong/etis-laribpolypdb'}
URLS = {'kvasir_instrument': ('https://datasets.simula.no/downloads/kvasir-instrument.zip', 'https://huggingface.co/datasets/medicalai/MedSegDB/resolve/main/Kvasir_Instrument.zip')}
URLS.update({
    'cvc_clinicdb': ('https://huggingface.co/datasets/medicalai/MedSegDB/resolve/main/CVC_ClinicDB.zip','https://www.kaggle.com/api/v1/datasets/download/orvile/cvc-clinicdb'),
    'cvc_colondb': ('https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing',),
    'etis': ('https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing',),
    'isic2018': ('https://huggingface.co/datasets/medicalai/MedSegDB/resolve/main/ISIC2018.zip',),
})
HF_BUS = 'MedOtter/BUS-UCLM'

def validate(root, name):
    dirs = _resolve_image_mask_dirs(root)
    if dirs is None: raise ValueError('No image/mask folders')
    pairs = _collect_pairs(dirs.image_dir, dirs.mask_dir)
    if len(pairs) != EXPECTED[name]: raise ValueError(f'{name}: expected {EXPECTED[name]} pairs, got {len(pairs)}')
    patients = set(); empty = 0
    for sid, image, mask in pairs:
        with Image.open(image) as im: im.load(); size = im.size
        binary = load_binary_mask(mask)
        if binary.size != size: raise ValueError(f'Image/mask dimension mismatch: {sid}')
        if not binary.getbbox(): empty += 1
        if name == 'bus_uclm':
            if not re.fullmatch(r'[a-z]{4}_\d{3}', sid): raise ValueError(f'Unexpected BUS patient/image ID: {sid}')
            patients.add(sid.split('_')[0])
    if name == 'bus_uclm' and (len(patients) != 38 or empty != 419):
        raise ValueError(f'BUS expected 38 patients and 419 empty masks, got {len(patients)}, {empty}')
    return pairs

def find_root(path, name):
    found = _find_dataset_root(path, name)
    candidates = ([found] if found else []) + [p for p in path.rglob('*') if p.is_dir()]
    for candidate in dict.fromkeys(candidates):
        if _resolve_image_mask_dirs(candidate) is None: continue
        try: validate(candidate, name); return candidate
        except (ValueError, RuntimeError, OSError): continue
    raise ValueError(f'No complete {name} layout in {path}')

def unpack(path, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for entry in z.infolist():
            if not (target / entry.filename).resolve().is_relative_to(target.resolve()):
                raise ValueError('Unsafe ZIP path')
        z.extractall(target)
    return target

def hf_bus(target):
    from datasets import load_dataset
    ds = load_dataset(HF_BUS, split='train')
    if len(ds) != 683: raise ValueError('Unexpected HF row count')
    (target/'images').mkdir(parents=True); (target/'masks').mkdir()
    classes = Counter(); seen = set()
    for row in ds:
        sid = canonical_sample_id(row['image_id'])
        if sid in seen or sid.split('_')[0] != str(row['patient_id']).lower(): raise ValueError('Invalid/duplicate patient ID')
        seen.add(sid); classes[str(row['class_label']).lower()] += 1
        row['image'].convert('RGB').save(target/'images'/f'{sid}.png')
        row['mask'].convert('L').save(target/'masks'/f'{sid}.png')
    if classes != {'normal':419,'benign':174,'malignant':90}: raise ValueError(f'Unexpected BUS classes: {classes}')
    return target

def acquire(name, cache):
    final = cache/name; receipt = cache/(name+'.json')
    cache.mkdir(parents=True, exist_ok=True)
    if final.is_dir() and receipt.is_file():
        try: validate(final,name); return json.loads(receipt.read_text())
        except Exception: shutil.rmtree(final); receipt.unlink(missing_ok=True)
    sources = []
    local = Path('/kaggle/input')
    if local.is_dir():
        for p in local.iterdir():
            key = re.sub(r'[^a-z0-9]','',p.name.lower())
            tokens={'kvasir_instrument':('kvasirinstrument',), 'bus_uclm':('busuclm',),
                    'cvc_clinicdb':('cvcclinicdb','clinicdb'), 'cvc_colondb':('cvccolondb','colondb'),
                    'etis':('etis',), 'isic2018':('isic2018',)}[name]
            if any(t in key for t in tokens):
                sources.append(('kaggle_input', str(p)))
    if name == 'bus_uclm': sources += [('hf_dataset', HF_BUS), ('https_zip','https://www.kaggle.com/api/v1/datasets/download/'+KAGGLE[name]), ('kagglehub', KAGGLE[name])]
    elif name == 'kvasir_instrument': sources += [('kagglehub', KAGGLE[name])] + [('https_zip', u) for u in URLS[name]] + [('https_zip','https://www.kaggle.com/api/v1/datasets/download/'+KAGGLE[name])]
    else:
        if name in KAGGLE:
            sources += [('https_zip','https://www.kaggle.com/api/v1/datasets/download/'+KAGGLE[name])]
        sources += [('https_zip', u) for u in URLS.get(name,())]
        if name in KAGGLE: sources += [('kagglehub', KAGGLE[name])]
        if name=='isic2018': sources += [('isic_official_pair','https://isic-challenge-data.s3.amazonaws.com/2018/')]

    errors = []
    for kind, source in sources:
        print('[ACQUIRE]',name,kind,source,flush=True)
        try:
            with tempfile.TemporaryDirectory(dir=cache) as td:
                tmp = Path(td)
                if kind == 'hf_dataset': root = hf_bus(tmp/'hf')
                elif kind == 'kagglehub':
                    import kagglehub
                    root = Path(kagglehub.dataset_download(source))
                elif kind == 'isic_official_pair':
                    root=tmp/'isic'; root.mkdir()
                    for label, filename in [('images','ISIC2018_Task1-2_Training_Input.zip'),('masks','ISIC2018_Task1_Training_GroundTruth.zip')]:
                        unpack(_maybe_download(source+filename,tmp/filename,verify=True),root/label)
                elif kind == 'https_zip': root = unpack(_maybe_download(source,tmp/'source.zip',verify=True),tmp/'extracted')
                else: root = Path(source)
                try: root = find_root(root,name)
                except ValueError:
                    archives = list(root.rglob('*.zip'))
                    if len(archives)!=1: raise
                    root = find_root(unpack(archives[0],tmp/'nested'),name)
                pairs = validate(root,name)
                staged = tmp/'canonical'; (staged/'images').mkdir(parents=True); (staged/'masks').mkdir()
                for sid, image, mask in pairs:
                    shutil.copy2(image,staged/'images'/f'{sid}{image.suffix.lower()}')
                    shutil.copy2(mask,staged/'masks'/f'{sid}{mask.suffix.lower()}')
                validate(staged,name)
                if final.exists(): shutil.rmtree(final)
                staged.rename(final)
                result = {'source':kind,'location':source,'path':str(final),'pairs':len(pairs),'failed_sources':errors}
                receipt.write_text(json.dumps(result,indent=2)); return result
        except Exception as e:
            errors.append(f'{kind} {source}: {e}'); print('[ACQUIRE FAILED]',errors[-1],flush=True)
    raise RuntimeError('All original-dataset sources failed: '+' | '.join(errors))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=EXPECTED,required=True); p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(acquire(a.dataset,a.cache)),flush=True)
if __name__=='__main__': main()
