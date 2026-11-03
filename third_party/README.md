# Third-party author implementations — intentionally NOT bundled

GFUNet: https://github.com/Zac2049/GFUNet (author implementation under `nnunet/`).
AFDSeg: https://github.com/promisedong/AFDSeg (author implementation).

This archive **does not contain a reimplemented or certified version** of either
method. Network access is needed to install exact, pinned upstream Git commits.
Use `scripts/revision/install_official_source.py` and complete the independent
paper-to-source audit before the strict adapter permits training.

Never relabel `external_spectral` or a small custom FFT model as a published
architecture. Do not claim that a Git clone alone verifies a paper baseline.

A strict adapter is available as model `verified_external_spectral` with explicit
`source_dir`, `expected_commit`, `audit_file`, `module_path`, `entrypoint`,
`model_kwargs`, `output_key`/`output_index`, and `output_type` fields. Select
these values **after inspecting the exact upstream source**. The adapter fails
closed when audits, sources, output specifications, or dependencies are missing.

For architectural/training faithfulness, upstream nnU-Net pipelines, auxiliary
losses, backbones and preprocessing must be preserved and documented. The
adapter does not automatically reproduce official preprocessing/training.
