# Kaggle AUTO downloader: Simula SSL certificate recovery

## Error fixed

The AUTO experiment queue failed at experiment 1 while downloading Kvasir-SEG:
`SSLCertVerificationError: unable to get local issuer certificate`.

## Behavior

- Attempt HTTPS with certificate verification enabled first.
- In AUTO mode, if and only if HTTPS fails with a certificate error at
  `datasets.simula.no`, retry the same archive without TLS certificate validation.
- Print a conspicuous warning; this fallback has a network interception risk.
- For non-Simula download hosts, certificate verification remains mandatory.
- The normal data preparation (image-mask pairing, resizing) still runs.
- Optional 1000-pair, SHA-256 and duplication audits remain disabled in AUTO mode.
- A failed experiment exits nonzero but does **not** advance the queue counter.

## Usage

1. Push the contents of this corrected repository to GitHub.
2. Use `BT_UNet_KAGGLE_11H_AUTO.ipynb` in Kaggle with GPU and Internet enabled.
3. Keep `FIRST_EXPERIMENT = 1` after the previous failed attempt.
4. Leave `PINNED_COMMIT = ''` for the first run on this fixed commit; pin the new SHA later.
5. The script writes `BT_UNET_NEXT_EXPERIMENT_AUTO.txt` and ZIPs completed jobs.

If the SSL recovery still fails because the official archive is unreachable,
check the job log. No experiment will be falsely marked completed.

### One-line emergency workaround for the old notebook

```python
import os
os.environ['ALLOW_INSECURE_DOWNLOAD'] = '1'
```

This disables certificate verification for all dataset downloads from the old
preparation script, not just Simula, so prefer the narrower revised version.

## Provenance caveat

The fallback does not guarantee data integrity against active interception.
For submission-quality work, consider a trusted source-provided checksum or
an independently authenticated mirror before treating the downloaded data as final.
