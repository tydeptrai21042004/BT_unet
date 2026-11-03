#!/usr/bin/env python3
"""Clone exact upstream source SHA without changing upstream files.

Requires Internet access on the user's own machine/Kaggle. An exact commit must
be supplied; this tool never guesses a moving branch's current commit. Cloning
is NOT itself paper-fidelity verification.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

SOURCES = {
    'gfunet': 'https://github.com/Zac2049/GFUNet.git',
    'afdseg': 'https://github.com/promisedong/AFDSeg.git',
}

def call(*args):
    subprocess.run(list(map(str,args)), check=True)

def install(name: str, commit: str, destination: Path) -> dict:
    if name not in SOURCES:
        raise ValueError(f'Unrecognized source: {name}')
    if len(commit) < 7 or not all(x in '0123456789abcdefABCDEF' for x in commit):
        raise ValueError('Provide an upstream Git commit SHA, not a branch or tag')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Not overwriting an existing directory: {destination}')
    call('git', 'clone', SOURCES[name], destination)
    call('git', '-C', destination, 'checkout', '--detach', commit)
    actual=subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip()
    if not actual.lower().startswith(commit.lower()):
        raise RuntimeError(f'Git revision mismatch: {actual} vs {commit}')
    manifest={'name':name,'official_repository':SOURCES[name], 'commit':actual,
              'source_dir':str(destination.resolve()),'status':'source_pinned_not_paper_verified',
              'warning':'Full architecture and training assumptions still require independent audit.'}
    (destination.parent/f'{name}_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',choices=sorted(SOURCES),required=True)
    p.add_argument('--commit',required=True,help='Exact upstream commit SHA (7-40 hex characters)')
    p.add_argument('--destination',required=True)
    args=p.parse_args()
    print(json.dumps(install(args.name,args.commit,Path(args.destination)),indent=2))
