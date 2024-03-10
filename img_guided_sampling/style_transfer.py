import argparse, os, sys
import torch
import torch.nn as nn
from PIL import Image

from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from guided_sampling.pipeline_sd_guided import StableDiffusionPipelineGuided

from transformers import CLIPTextModel, CLIPTokenizer
from clip.base_clip import CLIPEncoder


def numpy_to_pil(img):
    img = img.detach()
    img = img.cpu().permute(0, 2, 3, 1).float().numpy()
    img = (img * 255).round().astype("uint8")
    img_list = [Image.fromarray(im) for im in img]
    return img_list

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="the number of sampling steps"
    )
    parser.add_argument(
        "--style_ref_img_path",
        type=str,
        nargs="?",
        default="./style_images/xingkong.jpg",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/style-samples"
    )
    parser.add_argument(
        "--num_pred_steps",
        type=int,
        nargs="?",
        help="the number of steps to predict the outputs in advance",
        default=10,
    )
    parser.add_argument(
        "--pred_type",
        type=str,
        nargs="?",
        help="the ODE functions that choose to predict the x_0",
        default='type-1',
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    
    args = parser.parse_args()
    generator = (None if args.seed is None else torch.Generator(device='cuda').manual_seed(args.seed))
    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)
    
    pipe = StableDiffusionPipelineGuided.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float32)
    scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    pipe.safety_checker = None
    pipe.to('cuda')
    
    # is_grad = [False]*100
    # repeats =[1]*100
    is_grad = [False]*30 + [True]*40 + [False]*30
    repeats = [1]*30 + [2]*40 + [1]*30
    
    image_encoder = CLIPEncoder(need_ref=True, ref_path=args.style_ref_img_path).cuda()
    image_encoder.requires_grad_(False)
    
    loss_fn = image_encoder.get_gram_matrix_residual
    
    image = pipe(prompt = args.prompt, num_inference_steps = args.num_inference_steps, loss_fn = loss_fn, is_grad = is_grad, repeats = repeats, num_pred_steps = args.num_pred_steps, pred_type ="type-2", generator=generator).images[0]
    
    image.save(os.path.join(args.outdir, f"result.png"))
        

if __name__ == "__main__":
    main()
