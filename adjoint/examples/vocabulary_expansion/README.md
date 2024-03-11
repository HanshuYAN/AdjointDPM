# Vocabulary expansion of Stable Diffusion

We use AdjointDPM under a fine-grained visual classification (FGVC) model as guidance to help the Stable Diffusion model generate specific breeds of animals.

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

### Examples on vocabulary expansion
We provide an example script in run.sh.
```
python voca_expansion.py \
    --output_dir './voca_expansion_dog/' \
    --pretrained_fgvc_path "./dog.pth.tar" \
    --prompt 'A Cairn, a type of dog' \
    --model_guidance_dataset 'dog' \
    --model_guidance_class 41 \
    --grad_scale 0.05 \
    --num_train_epochs 30 \
    --learning_rate 0.01 
```
