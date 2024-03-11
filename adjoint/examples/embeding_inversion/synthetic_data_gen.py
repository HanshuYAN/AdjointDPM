import os
import torch
import argparse
import PIL
from PIL import Image
import sys
sys.path.append("src")
from src.diffusers import  StableDiffusionNODEPipelineForTrain, NeuralODEScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

def numpy_to_pil(img):
    img = img.detach()
    img = img.cpu().permute(0, 2, 3, 1).float().numpy()
    img = (img * 255).round().astype("uint8")
    img_list = [Image.fromarray(im) for im in img]
    return img_list

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a script to generate data.")

    parser.add_argument("--initial_prompt", type=str, required=True, help='Initial prompt')
    parser.add_argument("--after_prompt", type=str, required=True, help='Prompt after adding visual effects')
    parser.add_argument("--num_inference_steps", type=str, default=50, help='The number of inference steps')
    parser.add_argument("--guidance_scale", type=str, default=7.5, help='Guidance scale')
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default = "CompVis/stable-diffusion-v1-4",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--save_path", type=str, required=True, help='Path to store generated images')
    parser.add_argument("--index", type=int, required=True, help='Index number of generated samples')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    #shape of latent variable
    shape = (1, 4, 64, 64)
    
    #load model
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    pipe = StableDiffusionNODEPipelineForTrain.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler = NeuralODEScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    pipe.to('cuda')

    #initial noise
    noise = torch.randn(shape).to('cuda')
    initial = pipe(args.initial_prompt, num_inference_steps = 50, guidance_scale=7.5, latents=noise)
    initial_img = numpy_to_pil(initial)
    after = pipe(args.after_prompt, num_inference_steps = 50, guidance_scale=7.5, latents=noise)
    after_img = numpy_to_pil(after)

    #save path
    noise_path = os.path.join(args.save_path, f'noise-{args.index}')
    initial_path = os.path.join(args.save_path, f'initial_img-{args.index}.jpg')
    initial_path1 = os.path.join(args.save_path, f'initial_img-{args.index}')
    after_path = os.path.join(args.save_path, f'after_img-{args.index}.jpg') #jpg version
    after_path1= os.path.join(args.save_path, f'after_img-{args.index}') #tensor version

    torch.save(noise, noise_path)
    torch.save(initial, initial_path1)
    torch.save(after, after_path1)
    after_img[0].save(after_path)
    initial_img[0].save(initial_path)

if __name__ == "__main__":
    main()