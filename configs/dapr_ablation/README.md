# DAPR U-Net component ablation

This directory contains only the final proposal method and direct controls for it.

- `dapr_residual_control`: residual Fourier control with the same amplitude--phase spectral block.
- `dapr_no_global_phase`: removes phase modulation.
- `dapr_no_global_amplitude`: removes amplitude modulation.
- `dapr_no_global_channel_mix`: removes spectral channel mixing.
- `dapr_unet`: final Direct Amplitude--Phase Reconstruction U-Net.

Older APDR and DAPR-BAF proposal variants are intentionally removed from the active repository scope.
