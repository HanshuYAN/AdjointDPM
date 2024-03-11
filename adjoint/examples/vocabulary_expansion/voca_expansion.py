#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

from torch.utils.data import Dataset
import torch.utils.checkpoint
import torch.nn.functional as F 
import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from pathlib import Path
import random
import os
import math
import logging
import argparse
import sys
import diffusers
import PIL
from PIL import Image
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
sys.path.append("./diffuser")
from src.diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    StableDiffusionNODEPipelineForTrain,
    NeuralODEScheduler,
    NeuralODESchedulerInput,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from PIL import Image
from packaging import version
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from fgvc_ws_dan_helpers.inception_bap import inception_v3_bap


# TODO: remove and import from diffusers.utils when the new version of diffusers is released


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default = "CompVis/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_fgvc_path",
        type=str,
        required=True,
        help="path to the FGVC models from https://github.com/wvinzh/WS_DAN_PyTorch#result"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="voca_expansion",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='prompt used to generate rare class',
    )
    parser.add_argument(
        "--model_guidance_dataset",
        type=str, 
        default ='dog',
        required=True,
        help="kwargs for model_guidance_type (these are used in the below code)",
    )
    parser.add_argument(
        "--model_guidance_class",
        type=int, 
        default = 110,
        required=True,
        help="kwargs for model_guidance_type (these are used in the below code)",
    )
    parser.add_argument(
        "--grad_scale",
        type = float,
        default = 1,
        required=True,
        help="grad scale",
    )
    parser.add_argument("--num_inference_steps", type=int, default=30, help="number of sampling steps") 
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=30,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Initial learning rate (after the pote tial warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class MakeCutouts(nn.Module):
    """
    boiler plate multicrop for model guidance
    https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py
    """
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)
    

def fgvc_loss_fn(dataset,
                 device,
                 fgvc_path,
                 class_idx=0,
                 use_cutouts=False,
                 num_cuts=64,
                 cut_power=0.6,
                 grad_scale=0,
                 ):
    """
    This loss function is from https://github.com/salesforce/DOODL/"
    
    Guide with FGVC model from https://github.com/wvinzh/WS_DAN_PyTorch
    
    dataset (str): 'aircraft' (FGVC-Aircraft), 'bird' (CUB), 'dog' (Stanford Dogs)
    """
    # Models take in high resolution
    target_size = 512
    
    ### number of classes in each dataset
    num_class_dict = {'aircraft':100,
                      'bird':200,
                      'dog':120}
    
    # Get network 
    net = inception_v3_bap(pretrained=True, aux_logits=False)
    in_features = net.fc_new.in_features
    new_linear = torch.nn.Linear(
        in_features=in_features, out_features=num_class_dict[dataset])
    net.fc_new = new_linear
    net = net.to(device)
    ckpt = torch.load(fgvc_path)
    sd = {k.replace('module.',''):v for k,v in ckpt['state_dict'].items()}
    net.load_state_dict(sd)
    net.eval()
    
    # expected transform/multicrop
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
    if use_cutouts:
        cut = MakeCutouts(cut_size=target_size,
                          cut_power=cut_power)
        
    # gets vae decode as input           
    def loss_fn(im_pix):
        # prep image
        if use_cutouts:
            x_var = cut(im_pix, num_cuts)
        else:
            x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var)
        
        # last output is actual classification output, basic recognition objective
        _, _, output = net(x_var)
        # target is class idx
        target = (class_idx * torch.ones(output.size(0))).to(output.device).long()
        # Simple cross entropy loss
        loss = torch.nn.functional.cross_entropy(output, target)
        return grad_scale * loss

    return loss_fn


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def numpy_to_pil(img):
    img = img.detach()
    img = img.cpu().permute(0, 2, 3, 1).float().numpy()
    img = (img * 255).round().astype("uint8")
    img_list = [Image.fromarray(im) for im in img]
    return img_list

def custom_preprocess(image, target_size=(224, 224)):
    # Resize the image while preserving gradient information
    resized_image = F.interpolate(image, size=target_size, mode="bilinear", align_corners=False)

    # Normalize the image with the mean and standard deviation values used by CLIP
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(image.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(image.device)
    normalized_image = (resized_image - mean) / std

    return normalized_image


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )
    
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    generator = StableDiffusionNODEPipelineForTrain.from_pretrained(args.pretrained_model_name_or_path)
    scheduler = NeuralODESchedulerInput.from_config(generator.scheduler.config)
    generator.scheduler = scheduler
    generator.safety_checker = None
    generator.to(accelerator.device)

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # Freeze vae and unet
    generator.vae.requires_grad_(False)
    generator.unet.requires_grad_(False)
    # Freeze all parameters in text encoder
    text_encoder.requires_grad_(False)


    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        generator.unet.train()
        text_encoder.gradient_checkpointing_enable()
        generator.unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            generator.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    generator.vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    generator.unet.to(accelerator.device, dtype=weight_dtype)


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )  

    # The shape of initial random noise
    height = args.resolution
    width = args.resolution
    noise_shape = (args.train_batch_size, generator.unet.in_channels, height // generator.vae_scale_factor, width // generator.vae_scale_factor)
    noise = torch.randn(noise_shape).to(accelerator.device)
    noise_save_path = os.path.join(args.output_dir, f"noise")
    noise.requires_grad = True
    torch.save(noise.cpu(), noise_save_path)            

    #Initialize the optimizer
    optimizer = torch.optim.AdamW(
        [noise],  
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    orig_norm = noise.norm().item()
    
    loss_fn = fgvc_loss_fn(args.model_guidance_dataset,
                            class_idx=args.model_guidance_class,
                            fgvc_path=args.pretrained_fgvc_path,
                            use_cutouts=True,
                            num_cuts=16,
                            cut_power=0.3,
                            grad_scale=args.grad_scale,
                            device=accelerator.device)
    
    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,    
    )

    # Prepare everything with our `accelerator`.
    noise, optimizer,  lr_scheduler = accelerator.prepare(noise, optimizer,  lr_scheduler)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("vocabulary expansion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step
            resume_step = resume_global_step % (args.gradient_accumulation_steps)
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    prev_grad = torch.zeros_like(noise)
    for epoch in range(first_epoch, args.num_train_epochs):
        generator.unet.train()    
        with accelerator.accumulate(noise):
            
            prompt = args.prompt
            img = generator(prompt,  num_inference_steps = args.num_inference_steps, guidance_scale=7.5, latents=noise)
        
            img = img.to(accelerator.device)
            
            #save the output images of every step
            img_save_path = os.path.join(args.output_dir, f"img-{global_step}.png")
            img_pil = numpy_to_pil(img)
            img_pil[0].save(img_save_path)

            loss = loss_fn(img)
            
            accelerator.backward(loss)
            
#             grad = -0.5 * noise.grad.data
#             perturb_grad_scale = 1e-3
#             clip_grad_val = 1e-2
#             grad = grad.clip(-clip_grad_val, clip_grad_val)
#             perturbation = perturb_grad_scale * torch.randn_like(noise)
            
#             mom = 0.9
#             b = mom * prev_grad + grad
#             prev_grad = b.clone()
            
#             noise.data = perturbation + grad + noise.data
            
#             noise.data = noise.data * orig_norm / noise.data.norm().item()
#             noise.grad.data.zero_()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        print('epoch: {},  total loss: {}'.format(epoch, loss))
        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
        logs = {"epoch": epoch, "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"learned_noise")
        torch.save(noise.detach().cpu(), save_path)
        img_save_path = os.path.join(args.output_dir, f"changed_img.png")
        img_pil = numpy_to_pil(img)
        img_pil[0].save(img_save_path)
        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
