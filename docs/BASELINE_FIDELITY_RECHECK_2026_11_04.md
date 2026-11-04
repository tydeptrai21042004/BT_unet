# Baseline re-verification and reproducibility guard

This audit uses the user-uploaded `BT_unet-main(20261104-145403).zip` as the
local code snapshot. GitHub itself cannot be fetched from this execution
container; Kaggle cells clone `https://github.com/tydeptrai21042004/BT_unet.git`
**at execution time** and record the actual commit.

## Confirmed limitations, not assumptions

1. `docs/BASELINE_IMPLEMENTATION_AUDIT.md` identifies the 11 baseline methods
   as paper-*aligned* implementations, not byte-identical official methods.
2. `configs/official_faithful/` is a project convention, **not a certificate**:
   several configurations request `backbone_pretrained: false`, while the
   original published methods often use pretrained backbones and different losses.
3. `OfficialRes2NetEncoder`, `OfficialPVTv2Backbone`, `OfficialHarDNetEncoder`
   and `OfficialResNet34Encoder` include channel projection layers. Reusing a
   backbone family alone does not make the complete decoder architecture official.
4. The generic external spectral adapter does not implement GFUNet or AFDSeg.
   Their official repositories must be cloned and independently audited.
5. **Fixed:** If a caller explicitly asks for a pretrained backbone but checkpoint
   loading fails, backbone construction now raises instead of continuing with
   random weights.

## Implementation categories

| Models | Status |
|---|---|
| U-Net | Basic architecture reimplementation; no single canonical PyTorch publication checkpoint |
| U-Net++, Attention U-Net | Paper-aligned architectural controls, *not* official-source equivalence checked |
| PraNet, ACSNet, HarDNet-MSEG, CFANet, Polyp-PVT, CaraNet, HSNet, ResUNet++ | Local paper-aligned variants; **do not claim original-source exactness** |
| GFUNet, AFDSeg | Authentic original GitHub sources identified, **not integrated into local registry** |

Execute `python scripts/revision/audit_baseline_fidelity.py --output baseline_audit.json --strict-structure`
to produce a source/configuration report. The script checks local component
presence and suspicious configuration choices; it expressly **does not** issue
an official-fidelity certificate. To make that claim, pin and audit the authors'
original repository, compare architecture/initialization/forward/loss/preprocess
and reproduce official checkpoint evaluation using matching data splits.

## Kaggle runner policy

All experimental Kaggle cells must clone the declared project repository.
Run manifests record the actual Git commit and source-file hashes. Each
specialized local baseline run requires explicit `BT_ALLOW_PAPER_ALIGNED=1`.
This only authorizes a *paper-aligned comparison*; it never renames it an
original-author reproduction. The separate original-source audit cells clone
identified upstream repositories and inventory files/commits, without training
or declaring false verification.
