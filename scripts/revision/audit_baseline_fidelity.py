#!/usr/bin/env python3
"""Audit BT_unet's local baseline implementations WITHOUT falsely certifying paper fidelity.

Uses AST + YAML, so this audit runs without CUDA, torchvision, or datasets. It checks
presence of named component attributes and reports training-protocol deviations.
Paper-architecture equivalence requires actual original-source comparison beyond
this program. It is never inferred from passing these checks.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
# Identified authors' public source repositories, not automatically verified snapshots.
SPECS={
 'unet': dict(code='unet.py', cls='UNet', modules=['encoder','decoder','seg_head'], repo=None, category='architecture_control'),
 'unetpp': dict(code='unetpp.py', cls='UNetPlusPlus', modules=['x01','x11','x21','x31','x02','x12','x22','x03','x13','x04'], repo='https://github.com/MrGiovanni/UNetPlusPlus.git', category='paper_aligned'),
 'attention_unet': dict(code='attention_unet.py', cls='AttentionUNet', modules=['dec3','dec2','dec1','dec0'], repo='https://github.com/ozan-oktay/Attention-Gated-Networks.git', category='paper_aligned'),
 'pranet': dict(code='pranet.py', cls='PraNet', modules=['rfb2_1','rfb3_1','rfb4_1','agg1','ra4','ra3','ra2'], repo='https://github.com/DengPingFan/PraNet.git', category='specialized_paper_aligned'),
 'acsnet': dict(code='acsnet.py', cls='ACSNet', modules=['gcm','lca3','lca2','lca1','lca0','asm3','asm2','asm1','asm0'], repo='https://github.com/ReaFly/ACSNet.git', category='specialized_paper_aligned'),
 'hardnet_mseg': dict(code='hardnet_mseg.py', cls='HarDNetMSEG', modules=['rfb2','rfb3','rfb4','decoder'], repo='https://github.com/james128333/HarDNet-MSEG.git', category='specialized_paper_aligned'),
 'cfanet': dict(code='cfanet.py', cls='CFANet', modules=['boundary','stream1_43','stream2_01','bam0','seg_head'], repo='https://github.com/taozh2017/CFANet.git', category='specialized_paper_aligned'),
 'polyp_pvt': dict(code='polyp_pvt.py', cls='PolypPVT', modules=['backbone','cfm','cim','sam'], repo='https://github.com/DengPingFan/Polyp-PVT.git', category='specialized_paper_aligned'),
 'caranet': dict(code='caranet.py', cls='CaraNet', modules=['cfp','rfb2','rfb3','rfb4','agg','ara4','ara3','ara2'], repo='https://github.com/AngeLouCN/CaraNet.git', category='specialized_paper_aligned'),
 'hsnet': dict(code='hsnet.py', cls='HSNet', modules=['cnn_encoder','transformer_encoder','csa1','hsc1','msp'], repo='https://github.com/baiboat/HSNet.git', category='specialized_paper_aligned'),
 'resunetpp': dict(code='resunetpp.py', cls='ResUNetPlusPlus', modules=['stem','aspp_bridge','bridge_se','seg_head'], repo='https://github.com/DebeshJha/ResUNetplusplus.git', category='specialized_paper_aligned'),
}
EXTERNAL={
 'gfunet':'https://github.com/Zac2049/GFUNet.git',
 'afdseg':'https://github.com/promisedong/AFDSeg.git',
}

def object_attrs(code: Path, cls: str):
    tree=ast.parse(code.read_text(encoding='utf-8'),filename=str(code))
    definitions=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls]
    if not definitions:return None
    out=set()
    for n in ast.walk(definitions[0]):
        if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self':
            out.add(n.attr)
    return out

def git_sha():
    proc=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
    return proc.stdout.strip() if proc.returncode==0 else None

def filehash(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

def config_info(p):
    if not p.is_file():return None
    # PyYAML is standard in the benchmark environment; fallback avoids making this
    # provenance-only tool depend on a freshly installed ML environment.
    try:
        import yaml
        return yaml.safe_load(p.read_text(encoding='utf-8'))
    except ImportError:
        data={};section=None
        for line in p.read_text().splitlines():
            if line and not line.startswith(' ') and line.rstrip().endswith(':'):
                section=line.strip()[:-1];data[section]={}
            elif section and ':' in line and line.startswith('  '):
                k,v=line.strip().split(':',1)
                data[section][k]= {'false':False,'true':True}.get(v.strip().lower(),v.strip())
        return data

def audit_one(name):
    s=SPECS[name];f=ROOT/'src/models/baselines'/s['code'];attrs=object_attrs(f,s['cls']) if f.is_file() else None
    missing=sorted(set(s['modules'])-set(attrs or []))
    configs={}
    for proto in ['fair','official_faithful']:
        cfg=config_info(ROOT/'configs'/proto/(name+'.yaml'))
        if cfg is None:
            configs[proto]={'present':False};continue
        mc=cfg.get('model',{}) or {};tc=cfg.get('train',{}) or {}
        configs[proto]={
           'present':True,'backbone_pretrained':mc.get('backbone_pretrained','not specified'),
           'backbone_impl':mc.get('backbone_impl','not specified'),
           'backbone_variant':mc.get('res2net_variant',mc.get('pvt_variant',mc.get('hardnet_arch','not specified'))),
           'optimizer':tc.get('optimizer','not specified'),'loss':tc.get('loss','not specified'),
           'aux_loss_weight':tc.get('aux_loss_weight','not specified'),
           'use_aux_outputs_loss':tc.get('use_aux_outputs_loss','not specified'),
           'epochs':tc.get('epochs','not specified'),
        }
    flagged=[]
    if missing:flagged.append('Missing expected local component attributes: '+', '.join(missing))
    if configs['official_faithful'].get('backbone_pretrained') is False:
        flagged.append('so-called official_faithful config uses randomly initialized backbone')
    if name in ['pranet','caranet','acsnet','hardnet_mseg','polyp_pvt','hsnet']:
        flagged.append('Backbone stage projection modules alter native feature widths; check original topology')
    if name=='cfanet':
        flagged.append('Local Res2NetLikeEncoder is not the official Res2Net author checkpoint')
    return dict(baseline=name,local_implementation=s['category'],file=str(f.relative_to(ROOT)),
        source_sha256=filehash(f),original_author_repo=s['repo'],
        structural_check='pass' if not missing else 'fail',required_local_attributes=s['modules'],
        missing_attributes=missing,configs=configs,issues=flagged,
        fidelity='NOT_VERIFIED_AGAINST_ORIGINAL_SOURCE',
        independent_original_weights_tested=False,published_training_protocol_reproduced=False)

def generate():
    return dict(repository='https://github.com/tydeptrai21042004/BT_unet.git',git_commit=git_sha(),
       note='Structural attribute checks are NOT proof of an exact original-paper implementation.',
       results=[audit_one(n) for n in SPECS],
       external=[dict(name=k,official_repository=v,status='NOT_IN_LOCAL_MODEL_REGISTRY__NO_FIDELITY_VERIFICATION') for k,v in EXTERNAL.items()])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default='outputs_revision/baseline_audit.json')
    ap.add_argument('--strict-structure',action='store_true',help='Nonzero exit if any expected local component is missing')
    args=ap.parse_args()
    report=generate();dst=Path(args.output);dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    print('BASELINE_SOURCE_AUDIT',dst)
    for r in report['results']:
        print('{:16} {:5} {:30} {}'.format(r['baseline'],r['structural_check'],r['fidelity'], '; '.join(r['issues'])))
    print('External spectral: GFUNet and AFDSeg NOT VERIFIED; no fake substitute.')
    if args.strict_structure and any(r['structural_check']=='fail' for r in report['results']):
        raise SystemExit(2)
if __name__=='__main__':main()
