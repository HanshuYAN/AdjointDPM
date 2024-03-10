## Official Repo for Towards Accurate Guided Diffusion Sampling through Symplectic Adjoint Method


## Setup

First, download and set up the repo. We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. 

```bash
conda env create -f environment.yml
conda activate SAG
```

Install symplectic torchdiffeq: 

```bash
cd symplectic-adjoint-method-beta
python setup.py install
```
