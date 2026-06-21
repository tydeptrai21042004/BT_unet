# DAPR U-Net hyperparameter sensitivity

This suite is recommended in addition to the component ablation because DAPR has Fourier-specific knobs that can affect the result and reviewer confidence.

It tests the default DAPR model against changes in expansion ratio, amplitude scale, phase bound, and Fourier dropout while keeping the same dataset, split, optimizer, loss, epoch count, and evaluation protocol.
