# DAPR U-Net

**DAPR U-Net** means **Direct Amplitude--Phase Reconstruction U-Net**.

The method keeps the ordinary U-Net encoder and decoder, but replaces the deepest feature using a learned complex Fourier reconstruction:

- learn an amplitude response;
- learn a phase response;
- optionally mix spectral channels;
- invert the transformed spectrum;
- pass the reconstructed bottleneck feature to the decoder.

The defining design choice is that DAPR does not add the original bottleneck feature back as a residual bypass.

## Model names

Main proposal:

```text
dapr_unet
```

Component controls:

```text
dapr_residual_control
dapr_no_global_phase
dapr_no_global_amplitude
dapr_no_global_channel_mix
```

Hyperparameter sensitivity:

```text
dapr_expansion_1_0
dapr_expansion_2_0
dapr_amp_scale_0_5
dapr_amp_scale_1_5
dapr_phase_pi_half
dapr_phase_2pi
dapr_dropout_0_0
```

## Recommended manuscript evidence

Use three pieces of evidence:

1. fair comparison against the eleven baselines;
2. component ablation showing the effect of direct reconstruction, amplitude, phase, and channel mixing;
3. compact hyperparameter sensitivity showing that the proposal is not tuned from a single arbitrary Fourier setting.
