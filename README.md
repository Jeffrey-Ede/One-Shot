# One-Shot

Repository for the upcoming [paper](upcoming url!) "One-Shot Exit Wavefunction Reconstruction". 

One-shot exit wavefunction with deep learning uses deep learning to recover phases of conventional transmission electron microscopy images by restricting the distribution of wavefunctions. 

<p align="center">
  <img src="single_examples_refined-2.png">
</p>

Phases output by a neural network for input amplitudes are similar to true phases for In1.7K2Se8Sn2.28 wavefunctions. 

# Noteable Files

In the wavefunctions directory, subdirectories numbered 1,2,3, ..., snapshot neural networks as they were developed. After ~20 initial experiments, architecture was kept almost uncharged for the GAN and direct prediction.

**19**: n=1, multiple materials
**39**: n=3, multiple materials

**24**: n=1, single material
**38**: n=3, single material

**34**: n=1, single material, generative adversarial network
**37**: n=3, single material, generative adversarial network

**40**: n=3, multiple materials, restricted simulation hyperparameters
