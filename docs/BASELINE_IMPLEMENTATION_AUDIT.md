# Baseline implementation audit

This audit checks the eleven retained manuscript baselines. The repository
contains paper-aligned reimplementations; it does not claim byte-for-byte
identity with every authors' original training repository. Fair-comparison
configs intentionally use one shared loss and optimization protocol, whereas
`configs/official_faithful` preserves paper-style auxiliary/boundary outputs.

| Baseline | Implementation level | Fair config | Faithful config | Defining modules | Forward | Parameters |
|---|---|---:|---:|---:|---:|---:|
| U-Net | paper-architecture reimplementation | PASS | PASS | PASS | PASS | 7,849,601 |
| U-Net++ | paper-architecture reimplementation | PASS | PASS | PASS | PASS | 9,159,780 |
| Attention U-Net | paper-architecture reimplementation | PASS | PASS | PASS | PASS | 7,981,365 |
| PraNet | paper-aligned reimplementation with official-compatible Res2Net backbone | PASS | PASS | PASS | PASS | 17,141,567 |
| ACSNet | paper-aligned reimplementation with official-compatible ResNet-34 backbone | PASS | PASS | PASS | PASS | 32,839,181 |
| HarDNet-MSEG | paper-aligned reimplementation with official-compatible HarDNet-68 backbone | PASS | PASS | PASS | PASS | 19,005,765 |
| CFANet | paper-aligned reimplementation with boundary and cross-feature modules | PASS | PASS | PASS | PASS | 4,756,374 |
| Polyp-PVT | paper-aligned reimplementation with official-compatible PVT-v2 backbone | PASS | PASS | PASS | PASS | 2,729,290 |
| CaraNet | paper-aligned reimplementation with official-compatible Res2Net backbone | PASS | PASS | PASS | PASS | 16,449,084 |
| HSNet | paper-aligned dual-backbone reimplementation | PASS | PASS | PASS | PASS | 17,059,851 |
| ResUNet++ | paper-architecture reimplementation | PASS | PASS | PASS | PASS | 9,549,365 |

## Result

PASS
