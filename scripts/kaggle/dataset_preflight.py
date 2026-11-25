#!/usr/bin/env python3
"""Fast, non-destructive Kaggle-side probe for the fixed AUTO dataset queue.

This reads four ZIP signature bytes via streaming HTTP. It does NOT download the
whole dataset, calculate hashes, or claim that training data were verified. A
successful probe can still be followed by a download/extraction failure; the
scheduler records such a failure and skips that dataset without pretending the
experiment was completed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.registry import get_dataset_spec

# Same-dataset mirror operated by the dataset's original institution.
# Don't substitute unrelated datasets, processed derivatives or synthetic data.
OFFICIAL_MIRRORS = {
    'kvasir_instrument': ('https://huggingface.co/datasets/medicalai/MedSegDB/resolve/main/Kvasir_Instrument.zip',),
    'hyper_kvasir_seg': (
        'https://huggingface.co/datasets/SimulaMet-HOST/HyperKvasir/resolve/main/hyper-kvasir-segmented-images.zip',
    ),
}
AUTO_DATASETS = ('kvasir_seg', 'kvasir_instrument', 'isbi2012', 'hyper_kvasir_seg', 'bus_uclm')


def candidate_urls(dataset: str):
    if dataset not in AUTO_DATASETS:
        raise ValueError(f'Unapproved AUTO dataset: {dataset}')
    spec = get_dataset_spec(dataset)
    urls = (spec.default_download_url,) + spec.official_download_urls + OFFICIAL_MIRRORS.get(dataset, ())
    return tuple(dict.fromkeys(u for u in urls if u and u.startswith('https://')))


def probe_url(url: str, *, connect_timeout=5, read_timeout=9, tls_fallback=True):
    """Return small metadata or raise. Never fetches an entire ZIP intentionally."""
    import requests
    import urllib3
    host = (urlsplit(url).hostname or '').lower()
    last = None
    for verify in ((True, False) if host == 'datasets.simula.no' and tls_fallback else (True,)):
        try:
            # Some public file hosts reject HEAD or don't implement Range; use
            # streaming GET, read only four bytes, then close the connection.
            if not verify:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            with requests.get(url, headers={
                'User-Agent':'BT-UNet-Kaggle-preflight/2.0', 'Range':'bytes=0-3',
                'Accept':'application/zip,application/octet-stream,*/*',
            }, timeout=(connect_timeout, read_timeout), stream=True,
                allow_redirects=True, verify=verify) as response:
                response.raise_for_status()
                redirected = urlsplit(response.url)
                if redirected.scheme != 'https':
                    raise ValueError('Redirect downgraded HTTPS to insecure HTTP')
                first = next(response.iter_content(chunk_size=4), b'')
                if len(first) < 4 or first[:4] != b'PK\x03\x04':
                    raise ValueError('First four bytes are not ZIP local-file signature; '
                                     'dataset page, access gate, or invalid archive')
                return {
                    'url': url, 'resolved_host': redirected.hostname,
                    'range_supported': response.status_code == 206,
                    'https_certificate_verified': verify,
                    'status_code': response.status_code,
                    'content_length_header': response.headers.get('Content-Length'),
                }
        except requests.exceptions.SSLError as error:
            last = error
            if verify and host == 'datasets.simula.no' and tls_fallback:
                print('[DATA PREFLIGHT SSL] Retrying Simula host only with relaxed '
                      'certificate validation (interception risk):', url, flush=True)
                continue
            break
        except (requests.exceptions.RequestException, ValueError) as error:
            last = error
            break
    raise RuntimeError(f'ZIP probe failed: {type(last).__name__}: {last}')


def check_sources(datasets=AUTO_DATASETS, *, max_total_seconds=110, probe=probe_url):
    started=time.monotonic()
    result={}
    for dataset in datasets:
        errors=[]
        selected=None
        for url in candidate_urls(dataset):
            if time.monotonic()-started > max_total_seconds:
                errors.append('Preflight overall deadline reached')
                break
            try:
                selected=probe(url)
                print('[PREFLIGHT OK]',dataset,url,
                      'TLS verified:',selected.get('https_certificate_verified'),flush=True)
                break
            except Exception as error:
                errors.append(f'{url} :: {error}')
                print('[PREFLIGHT UNAVAILABLE]',dataset,str(error)[:180],flush=True)
        result[dataset] = {
            'available': selected is not None if candidate_urls(dataset) else None,
            'selected': selected,
            'attempted_urls': list(candidate_urls(dataset)),
            'errors': errors,
        }
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets',nargs='+',default=list(AUTO_DATASETS),choices=AUTO_DATASETS)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seconds',type=int,default=110)
    ns=ap.parse_args()
    result={'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'probe_only_not_full_dataset_download':True,
            'datasets':check_sources(ns.datasets,max_total_seconds=ns.seconds)}
    ns.output.parent.mkdir(parents=True,exist_ok=True)
    ns.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('[AUTO PREFLIGHT SUMMARY]',
          {k:('SDK acquisition' if v['available'] is None else 'OK' if v['available'] else 'probe failed; acquisition may still succeed') for k,v in result['datasets'].items()},flush=True)
    # Unavailable data is not a program failure. The runner will skip all related
    # experiments and retain their IDs in the skipped-jobs manifest.
    return 0

if __name__=='__main__':
    raise SystemExit(main())
