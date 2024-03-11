import os
import torch

import PIL
from PIL import Image
import sys
sys.path.append("src")
from diffusers import StableDiffusionNODEPipeline, NeuralODEScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

def numpy_to_pil(img):
    img = img.detach()
    img = img.cpu().permute(0, 2, 3, 1).float().numpy()
    img = (img * 255).round().astype("uint8")
    img_list = [Image.fromarray(im) for im in img]
    return img_list

num_checkpoints = 80

initial_prompt = 'a  photograph of a cottage by the lake in icelandic country side'
after_prompt = 'a  photograph of a cottage by the lake in icelandic country side'
shape = (1, 4, 64, 64)
#load model
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
generator = StableDiffusionNODEPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
scheduler = NeuralODEScheduler.from_config(generator.scheduler.config)
generator.scheduler = scheduler
generator.safety_checker = None
generator.to('cuda')


#initial_prompt embedding
text_token_part = tokenizer(initial_prompt, return_tensors='pt').input_ids
encoder_hidden_states_part = text_encoder(text_token_part)[0].detach().to('cuda')


path = "/home/tiger/diffusion/diffuser/inverse_prompt_scene"
noise = torch.randn(shape).to('cuda')
final_path = os.path.join(path, f'final_new2.jpg')
final_embeds = torch.load("/home/tiger/diffusion/diffuser/inverse_prompt_scene/learned_embeds").to('cuda')
final_hidden_states = torch.cat([encoder_hidden_states_part, final_embeds], dim = 1).to('cuda')
final_img = generator(num_inference_steps = 50, guidance_scale=7.5, latents=noise, prompt_embeds=final_hidden_states).images[0]
final_img.save(final_path)

initial_path = os.path.join(path, f'initial_img_new2.jpg')
after_path = os.path.join(path, f'after_img_new2.jpg')
initial_img = generator(initial_prompt, num_inference_steps = 50, guidance_scale=7.5, latents=noise).images[0]
after_img = generator(after_prompt, num_inference_steps = 50, guidance_scale=7.5, latents=noise).images[0]
after_img.save(after_path)
initial_img.save(initial_path)

# for i in range(0, num_checkpoints, 5):
#     print(i)
    
#     trained_path = os.path.join(path, f'checkpoint-{i}')
#     trained_embeds = torch.load(trained_path).to('cuda')

#     # noise_path = "/home/tiger/assets/ReferImagesPrompt/noise-1"
#     # noise = torch.load(noise_path)
    
#     encoder_hidden_states = torch.cat([encoder_hidden_states_part, trained_embeds], dim = 1).to('cuda')
#     img = generator(num_inference_steps = 50, guidance_scale=7.5, latents=noise, prompt_embeds=encoder_hidden_states).images[0]

#     save_path = os.path.join(path, f'img_new1-{i}.jpg')
#     img.save(save_path)

