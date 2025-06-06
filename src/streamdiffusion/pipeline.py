import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from streamdiffusion.image_filter import SimilarImageFilter

class StreamDiffusion:
    def __init__(
        self,
        pipe: DiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (
                    self.denoising_steps_num + 1
                ) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = (
                    2 * self.denoising_steps_num * self.frame_bff_size
                )
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae
        
        self.inference_time_ema = 0

        self.sdxl = type(self.pipe) is StableDiffusionXLPipeline

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        if self.sdxl:
            self.add_text_embeds = encoder_output[2]
            original_size = (self.height, self.width)
            crops_coords_top_left = (0, 0)
            target_size = (self.height, self.width)
            text_encoder_projection_dim = int(self.add_text_embeds.shape[-1])
            self.add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=encoder_output[0].dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )

        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            self.prompt_embeds = torch.cat(
                [uncond_prompt_embeds, self.prompt_embeds], dim=0
            )

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = (
            self.alpha_prod_t_sqrt[t_index] * original_samples
            + self.beta_prod_t_sqrt[t_index] * noise
        )
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # Debug inputs to scheduler step
        try:
            print(f"[SDXL DEBUG] scheduler_step_batch input: shape={x_t_latent_batch.shape}, dtype={x_t_latent_batch.dtype}")
            print(f"[SDXL DEBUG] model_pred: shape={model_pred_batch.shape}, min={model_pred_batch.min().item():.4f}, max={model_pred_batch.max().item():.4f}")
            
            # Check for NaNs in inputs
            if torch.isnan(model_pred_batch).any() or torch.isinf(model_pred_batch).any():
                print(f"[SDXL DEBUG] WARNING: model_pred contains {torch.isnan(model_pred_batch).sum().item()} NaN values")
                model_pred_batch = torch.nan_to_num(model_pred_batch, nan=0.0, posinf=1.0, neginf=-1.0)
                
            if torch.isnan(x_t_latent_batch).any() or torch.isinf(x_t_latent_batch).any():
                print(f"[SDXL DEBUG] WARNING: x_t_latent_batch contains {torch.isnan(x_t_latent_batch).sum().item()} NaN values")
                x_t_latent_batch = torch.nan_to_num(x_t_latent_batch, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception as e:
            print(f"[SDXL DEBUG] Error checking scheduler inputs: {e}")
            
        # TODO: use t_list to select beta_prod_t_sqrt
        try:
            if idx is None:
                F_theta = (
                    x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
                ) / self.alpha_prod_t_sqrt
                denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
            else:
                F_theta = (
                    x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
                ) / self.alpha_prod_t_sqrt[idx]
                denoised_batch = (
                    self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
                )
                
            # Check for NaNs in output
            if torch.isnan(denoised_batch).any() or torch.isinf(denoised_batch).any():
                print(f"[SDXL DEBUG] WARNING: scheduler output contains {torch.isnan(denoised_batch).sum().item()} NaN and {torch.isinf(denoised_batch).sum().item()} inf values")
                denoised_batch = torch.nan_to_num(denoised_batch, nan=0.0, posinf=1.0, neginf=-1.0)
                
            print(f"[SDXL DEBUG] scheduler output: min={denoised_batch.min().item():.4f}, max={denoised_batch.max().item():.4f}, mean={denoised_batch.mean().item():.4f}")
            return denoised_batch
        except Exception as e:
            print(f"[SDXL DEBUG] Error in scheduler calculation: {e}")
            raise

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        added_cond_kwargs, 
        idx: Optional[int] = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Debug input latent
        print(f"[SDXL DEBUG] unet_step input: shape={x_t_latent.shape}, dtype={x_t_latent.dtype}, t_indices={t_list}")
        try:
            print(f"[SDXL DEBUG] unet_step input range: min={x_t_latent.min().item():.4f}, max={x_t_latent.max().item():.4f}, mean={x_t_latent.mean().item():.4f}")
            if torch.isnan(x_t_latent).any() or torch.isinf(x_t_latent).any():
                print(f"[SDXL DEBUG] WARNING: Input to UNet contains {torch.isnan(x_t_latent).sum().item()} NaN and {torch.isinf(x_t_latent).sum().item()} inf values")
                x_t_latent = torch.nan_to_num(x_t_latent, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception as e:
            print(f"[SDXL DEBUG] Error checking UNet input: {e}")
        
        # Prepare inputs based on CFG type
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        # Run UNet prediction
        print(f"[SDXL DEBUG] Running UNet with input shape={x_t_latent_plus_uc.shape}, timesteps={t_list}")
        try:
            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # Check UNet output
            print(f"[SDXL DEBUG] UNet output: shape={model_pred.shape}, dtype={model_pred.dtype}")
            print(f"[SDXL DEBUG] UNet output range: min={model_pred.min().item():.4f}, max={model_pred.max().item():.4f}, mean={model_pred.mean().item():.4f}")
            if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                print(f"[SDXL DEBUG] WARNING: UNet output contains {torch.isnan(model_pred).sum().item()} NaN and {torch.isinf(model_pred).sum().item()} inf values")
                model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1.0, neginf=-1.0)
        except Exception as e:
            print(f"[SDXL DEBUG] Error in UNet prediction: {e}")
            raise
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [self.init_noise[1:], self.init_noise[0:1]], dim=0
                )
                self.stock_noise = init_noise + delta_x

        else:
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred
        
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        print(f"[SDXL DEBUG] encode_image input: shape={image.shape}, dtype={image.dtype}, range=[{image.min().item():.4f}, {image.max().item():.4f}], mean={image.mean().item():.4f}")
        
        # First convert to device with model's default dtype
        image = image.to(device=self.device, dtype=self.dtype)
        
        # Check for NaNs in input image
        if torch.isnan(image).any():
            print(f"[SDXL DEBUG] WARNING: Input image contains {torch.isnan(image).sum().item()} NaN values")
            image = torch.nan_to_num(image, nan=0.0)
        
        # ALWAYS use FP32 for VAE operations to prevent NaN issues
        # Regardless of the model's overall dtype (self.dtype)
        image = image.to(torch.float32)
        print(f"[SDXL DEBUG] Converted input to float32 for VAE encoding to prevent NaNs, dtype={image.dtype}")
            
        try:
            # VAE encode
            latent = self.vae.encode(image).latent_dist.sample(generator=self.generator)
            latent = self.vae.config.scaling_factor * latent
            
            # Check for NaNs in latent
            if torch.isnan(latent).any():
                print(f"[SDXL DEBUG] WARNING: VAE encoder output contains {torch.isnan(latent).sum().item()} NaN values")
                latent = torch.nan_to_num(latent, nan=0.0)
                
            if latent.dtype != self.dtype:
                latent = latent.to(self.dtype)
                
            print(f"[SDXL DEBUG] encode_image output latent: shape={latent.shape}, dtype={latent.dtype}, range=[{latent.min().item():.4f}, {latent.max().item():.4f}], mean={latent.mean().item():.4f}")
            return latent
        except Exception as e:
            print(f"[SDXL DEBUG] Error in VAE encoding: {e}")
            raise

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:    
        # Log input latent before VAE decode
        print(f"[SDXL DEBUG] decode_image input latent: shape={x_0_pred_out.shape}, dtype={x_0_pred_out.dtype}, range=[{x_0_pred_out.min().item():.4f}, {x_0_pred_out.max().item():.4f}], mean={x_0_pred_out.mean().item():.4f}")
        
        # Check for NaNs in input latent
        if torch.isnan(x_0_pred_out).any() or torch.isinf(x_0_pred_out).any():
            print(f"[SDXL DEBUG] WARNING: Input latent to decoder contains {torch.isnan(x_0_pred_out).sum().item()} NaN values and {torch.isinf(x_0_pred_out).sum().item()} inf values")
            x_0_pred_out = torch.nan_to_num(x_0_pred_out, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            # ALWAYS use FP32 for VAE operations to prevent NaN issues
            x_0_pred_out = x_0_pred_out.to(torch.float32)
            print(f"[SDXL DEBUG] Converted latent to float32 for VAE decoding to prevent NaNs, dtype={x_0_pred_out.dtype}")
            
            # Scale and decode
            scaled_latent = x_0_pred_out / self.vae.config.scaling_factor
            output_latent = self.vae.decode(
                scaled_latent,
                return_dict=False,
            )[0]
            
            # Check for NaNs in output image
            if torch.isnan(output_latent).any() or torch.isinf(output_latent).any():
                print(f"[SDXL DEBUG] WARNING: VAE decoder output contains {torch.isnan(output_latent).sum().item()} NaN values and {torch.isinf(output_latent).sum().item()} inf values")
                output_latent = torch.nan_to_num(output_latent, nan=0.0, posinf=1.0, neginf=-1.0)
                
            print(f"[SDXL DEBUG] decode_image output: shape={output_latent.shape}, dtype={output_latent.dtype}, range=[{output_latent.min().item():.4f}, {output_latent.max().item():.4f}], mean={output_latent.mean().item():.4f}")
            return output_latent
            
        except Exception as e:
            print(f"[SDXL DEBUG] Error in VAE decoding: {e}")
            raise

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        added_cond_kwargs = {}
        prev_latent_batch = self.x_t_latent_buffer
        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                self.stock_noise = torch.cat(
                    (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                )
            if self.sdxl:
                added_cond_kwargs = {"text_embeds": self.add_text_embeds.to(self.device), "time_ids": self.add_time_ids.to(self.device)}

            x_t_latent = x_t_latent.to(self.device)
            t_list = t_list.to(self.device)
            x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list, added_cond_kwargs=added_cond_kwargs)
            
            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                if self.sdxl:
                    added_cond_kwargs = {"text_embeds": self.add_text_embeds.to(self.device), "time_ids": self.add_time_ids.to(self.device)}
                x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx=idx, added_cond_kwargs=added_cond_kwargs)
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[
                            idx + 1
                        ] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(
                            x_0_pred, device=self.device, dtype=self.dtype
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred
        return x_0_pred_out

    @torch.no_grad()
    def __call__(
        self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None
    ) -> torch.Tensor:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if x is not None:
            x = self.image_processor.preprocess(x, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
            if self.similar_image_filter:
                x = self.similar_filter(x)
                if x is None:
                    time.sleep(self.inference_time_ema)
                    return self.prev_image_result
            x_t_latent = self.encode_image(x)
        else:
            # TODO: check the dimension of x_t_latent
            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        x_output = self.decode_image(x_0_pred_out).detach().clone()

        self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        x_0_pred_out = self.predict_x0_batch(
            torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        )
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        return x_output

    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )
        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            encoder_hidden_states=self.prompt_embeds,
            return_dict=False,
        )[0]
        x_0_pred_out = (
            x_t_latent - self.beta_prod_t_sqrt * model_pred
        ) / self.alpha_prod_t_sqrt
        return self.decode_image(x_0_pred_out)