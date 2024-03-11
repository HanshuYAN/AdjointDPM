## Official Repo for 
- ## AdjointDPM: Adjoint Sensitivity Method for Gradient Backpropagation of Diffusion Probabilistic Models (ICLR 2024)
- ## Towards Accurate Guided Diffusion Sampling through Symplectic Adjoint Method

### [Paper1](https://openreview.net/pdf?id=y33lDRBgWI) | [Paper2](https://arxiv.org/pdf/2312.12030.pdf) | [Project Page] 


AdjointDPM is a method that can not only finetune the parameters of DPMs, including network parameters and text embedding, but also perform guided sampling with accurate gradient guidance, based on a differentiable metric defined on the generated contents. 

There are several interesting experiments to demonstrate the effectiveness of AdjointDPM. For the finetuning tasks, including stylization and text embedding inversion, they are implemented based on [🧨 Diffusers](https://github.com/huggingface/diffusers). For the guided sampling, they are implemented in [img_guided_sampling](https://github.com/HanshuYAN/AdjointDPM/tree/main/img_guided_sampling). For security auditing under an ImageNet classifier, we implement the code heavily based on [dpm-solver](https://github.com/LuChengTHU/dpm-solver/tree/main/examples/ddpm_and_guided-diffusion) codebase. Check it in [ddpm_and_guided-diffusion](https://github.com/HanshuYAN/AdjointDPM/tree/main/ddpm_and_guided-diffusion). 


## Setup

First, download and set up the repo:

```bash
git clone https://github.com/HanshuYAN/AdjointDPM.git
cd AdjointDPM
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. 

```bash
conda env create -f environment.yml
conda activate adjointDPM
```
