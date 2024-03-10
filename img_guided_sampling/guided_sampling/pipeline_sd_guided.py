from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from torch_symplectic_adjoint import odeint_symplectic_adjoint as odeint

# from ...configuration_utils import FrozenDict
# from ...models import AutoencoderKL, UNet2DConditionModel
# from ...schedulers import KarrasDiffusionSchedulers
# from ..pipeline_utils import DiffusionPipeline
# from .safety_checker import StableDiffusionSafetyChecker

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import logging, replace_example_docstring

# from ...utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
# from . import StableDiffusionPipelineOutput
# from .pipeline_stable_diffusion import StableDiffusionPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableDiffusionPipelineGuided(StableDiffusionPipeline):
    """_summary_

    Args:
        StableDiffusionPipeline (_type_): _description_
    """
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        #image = image.cpu().permute(0, 2, 3, 1).float().numpy() #!!!:
        return image
    
    def node_solver_adaptive(self, 
                    model_input, 
                    index, 
                    num_steps=10,
                    guidance_scale=1.,
                    prompt_embeds=None,
                    solver_type="euler",
                    cross_attention_kwargs=None,         
                   ):
        
        timesteps = self.scheduler.timesteps
        alphas_cumprod = self.scheduler.alphas_cumprod
        sigmas = np.array(((1 - alphas_cumprod.cpu()) / alphas_cumprod.cpu()) ** 0.5)
        sigmas = np.interp(np.array(timesteps.cpu()), np.arange(0, len(sigmas)), sigmas)
        
        timesteps = timesteps[index:]
        sigmas = torch.from_numpy(sigmas).to(timesteps.device)[index:]
        if timesteps.shape[0] >= num_steps:
            indices_to_extract = torch.linspace(0, len(timesteps)-1, num_steps, dtype=torch.long)
            timesteps_short = timesteps[indices_to_extract]
            sigmas_short = sigmas[indices_to_extract]
        else:
            timesteps_short = timesteps
            sigmas_short = sigmas
            
        init_sigmas =((sigmas_short[0].item() ** 2 + 1) ** 0.5)
        
        if num_steps == 1:
            sigmas_short = torch.cat([sigmas_short, torch.tensor([0.0], device=sigmas_short.device, dtype=sigmas_short.dtype)])
            timesteps_short = torch.cat([timesteps_short, torch.tensor([0], device=timesteps_short.device, dtype=timesteps_short.dtype)])

        def ODEFunc(t, x):
            
            t_index = torch.argmin(torch.abs(sigmas_short - t))
            ts = timesteps_short[t_index].unsqueeze(0)
            delta=sigmas_short[t_index]
            latent_input = x / ((delta.item() ** 2 + 1) ** 0.5)
            x_in = torch.cat([latent_input] * 2)
            noise_pred = self.unet(x_in, ts, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs).sample
            e_t_uncond, e_t = noise_pred.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            
            return e_t
        
        solution = odeint(ODEFunc, model_input * init_sigmas, sigmas_short, method=solver_type)
        
        pred_x0 = solution[-1]
        
        return pred_x0

    def node_solver_adaptive_type2(self, 
                    model_input, 
                    index, 
                    num_steps=10,
                    guidance_scale=1.,
                    prompt_embeds=None,
                    solver_type="euler",
                    cross_attention_kwargs=None,         
                   ):
        
        timesteps = self.scheduler.timesteps
        alphas_cumprod = self.scheduler.alphas_cumprod
        sigmas = np.array((alphas_cumprod.cpu() / (1 - alphas_cumprod.cpu())) ** 0.5)
        sigmas = np.interp(np.array(timesteps.cpu()), np.arange(0, len(sigmas)), sigmas)
        
        timesteps = timesteps[index:]
        sigmas = torch.from_numpy(sigmas).to(timesteps.device)[index:]
        if timesteps.shape[0] >= num_steps:
            indices_to_extract = torch.linspace(0, len(timesteps)-1, num_steps, dtype=torch.long)
            timesteps_short = timesteps[indices_to_extract]
            sigmas_short = sigmas[indices_to_extract]
        else:
            timesteps_short = timesteps
            sigmas_short = sigmas
            
        init_sigmas =((sigmas_short[0].item() ** 2 + 1) ** 0.5)
        
        def ODEFunc(t, x):
            
            t_index = torch.argmin(torch.abs(sigmas_short - t))
            ts = timesteps_short[t_index].unsqueeze(0)
            delta = sigmas_short[t_index]
            sqrt_alpha = delta.item() / ((delta.item() ** 2 + 1) ** 0.5)
            sqrt_one_minus_alpha = 1 / ((delta.item() ** 2 + 1) ** 0.5)
            
            latent_input = x / ((delta.item() ** 2 + 1) ** 0.5)
            x_in = torch.cat([latent_input] * 2)
            noise_pred = self.unet(x_in, ts, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs).sample
            e_t_uncond, e_t = noise_pred.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            
            func = (latent_input - sqrt_one_minus_alpha * e_t) / sqrt_alpha
            
            return func
        
        solution = odeint(ODEFunc, model_input * init_sigmas, sigmas_short, method=solver_type)
        
        pred_x0 = solution[-1] / ((sigmas_short[-1].item() ** 2 + 1) ** 0.5)
        
        return pred_x0
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # guidance args
        loss_fn = None,
        is_grad: List[bool] = None,
        repeats: List[int] = None,
        num_pred_steps: int = 10, 
        pred_type: str = "type-2",
        solver_type: str = "euler",
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            loss_fn: 
                loss function defined on the output of stable diffusion models.
            is_grad ('List[bool]', *optional*, defaults to [True] * num_inference_steps):
                The list indicates whether the steps needs to be modified by the gradient information.
            repeats ('List[int]', *optional*, defaults to [1] * num_inference_steps):
                The list indicates the number of repeats at each sampling step to do time-travel strategy.
            num_pred_steps ('int', *optional*, defaults to 10):
                The number of steps used to predict the x_0 during the sampling process.
            pred_type ('str', *optional*, defaults to "type-2"):
                The different form of ODE functions.
            solver_type ('str', *optional*, defaults to "euler"):
                The methods to solve neural ODE functions.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        if is_grad == None:
            is_grad = [True] * num_inference_steps
        
        if repeats == None:
            repeats = [1] * num_inference_steps
        

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        alphas_cumprod = self.scheduler.alphas_cumprod
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                repeat = repeats[i]
                x =  latents.detach().requires_grad_(True)
                for j in range(repeat):
                    x = x.detach().requires_grad_(True)
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([x] * 2) if do_classifier_free_guidance else x
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        correction = noise_pred_text - noise_pred_uncond
                    
                    # compute the previous noisy sample x_t -> x_t-1
                    x_prev = self.scheduler.step(noise_pred, t, x, **extra_step_kwargs).prev_sample
                    
                    #>>> !!!: Guided sampling >>>> ###
                    with torch.enable_grad():
                        if is_grad[i]:
                            if pred_type == "type-1":
                                pred_x0 = self.node_solver_adaptive(x, i, num_pred_steps, guidance_scale, prompt_embeds, solver_type, cross_attention_kwargs)
                            elif pred_type == 'type-2':
                                pred_x0 = self.node_solver_adaptive_type2(x, i, num_pred_steps, guidance_scale, prompt_embeds, solver_type, cross_attention_kwargs)
                            D_x0_t = self.decode_latents(pred_x0)

                            # intermediate = D_x0_t.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                            # intermediate = self.numpy_to_pil(intermediate)
                            # intermediate[0].save(os.path.join("./outputs/style_1/", "int_{}.png".format(i)))
                            
                            residual = loss_fn(D_x0_t)
                            norm = torch.linalg.norm(residual)
                            norm.backward()
                            norm_grad = x.grad.data
                            
                            rho = guidance_scale * 0.2 \
                                * (correction * correction).mean().sqrt().item() \
                                / (norm_grad * norm_grad).mean().sqrt().item()
                                
                            x_prev = x_prev - rho * norm_grad.detach()
                        
                    if i < len(timesteps) -1:
                        a_t = alphas_cumprod[t]
                        t_prev = timesteps[i + 1]
                        a_prev = alphas_cumprod[t_prev]
                        beta_t = a_t / a_prev
                        
                        noise = torch.randn_like(x_prev)
                        x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise
                    ### <<<< !!!: Guided sampling <<<#
                    
                latents = x_prev.detach()
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy() # since not implemented in func-decode

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
