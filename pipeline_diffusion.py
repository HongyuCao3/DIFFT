import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import Transformer2DModel

from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict

# from diffusers.loaders import FromCkptMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
from tqdm import tqdm
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class DiffTipeline(StableDiffusionPipeline):
    def __init__(
        self, reward_model, scheduler: DDIMScheduler, target=1, target_guidance=100
    ):  # ,scheduler: DDIMScheduler
        self.scheduler = scheduler
        self.target = target
        self.target_guidance = target_guidance
        self.reward_model = reward_model
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()

    @torch.enable_grad()
    def compute_gradient(self, latents, target=None, step=None):
        """计算基于奖励模型输出与目标之间的二次误差的梯度，并返回对潜变量的梯度与模型输出

        此方法将输入的潜变量张量重排至 (T, B, C) 形状，启用梯度，并通过 reward_model.evaluate 计算输出。
        以 0.5 * MSELoss(out, target) 为目标函数，对潜变量执行反向传播，返回对应的梯度副本与前向输出。
        注意：该函数会在反向传播前对奖励模型的梯度进行清零（zero_grad），并触发一次 backward 图计算。

        Args:
         - latents (torch.Tensor): 输入潜变量张量，原始期望形状为 (B, T, C)。内部将被重排为 (T, B, C) 并启用 requires_grad。
         - target (torch.Tensor, optional): 目标张量，形状为 (B, 1)。若为 None，则使用实例属性 self.target（标量）在对应设备上构造并按批大小重复。
         - step (int, optional): 当前步数，仅用于调试或日志记录，不影响计算。

        Returns: Tuple[torch.Tensor, torch.Tensor]: - torch.Tensor: 相对于重排后潜变量 (T, B, C) 的梯度，形状与重排后的 latents 一致。 - torch.Tensor: 奖励模型的前向输出，一般形状为 (B, 1)，与 target 对齐。

        Raises: RuntimeError: 当张量形状不匹配、设备不一致或反向传播图构建失败时可能抛出。"""
        if target is None:
            target = torch.FloatTensor([[self.target]]).to(latents.device)
            target = target.repeat(latents.shape[0], 1)

        latents = latents.permute(1, 0, 2).contiguous()  # 705,B,C
        latents.requires_grad_(True)
        out = self.reward_model.evaluate(latents)
        # print('setp_',step,':',out[0].item())
        l2_error = 0.5 * torch.nn.MSELoss()(out, target)
        self.reward_model.zero_grad()
        l2_error.backward()
        return latents.grad.clone(), out

    @torch.no_grad()
    def __call__(
        self,
        diffusion_model: torch.nn.Module,
        shape: Union[List[int], Tuple[int]],
        cond: torch.FloatTensor,
        steps: int,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        use_reward: int = 0,
        reward_target=None,
        target=1,
        guidance=1,
        generator: Optional[torch.Generator] = None,
        device: torch.device = "cuda:0",
        disable_prog: bool = True,
    ):
        """ 执行扩散采样流程（DDIM），支持无分类器引导（CFG）与可选的奖励引导梯度修正

        该方法以给定的扩散模型与条件向量为输入，从高斯噪声初始化潜变量并按调度器时间步迭代更新，最终返回生成的潜变量以及各步的中间结果。若启用奖励引导，将在每步对预测噪声进行基于外部评价的梯度修正。

        Args: 
         - diffusion_model (torch.nn.Module): 噪声预测模型/扩散模型，需接受 (latent, timestep, cond) 形式的输入并输出噪声残差。 - shape (Union[List[int], Tuple[int]]): 初始潜变量的张量形状。 
         - cond (torch.FloatTensor): 条件向量（例如文本/图像嵌入）。当启用无分类器引导（guidance_scale > 1.0）时，内部会自动拼接对应的无条件向量。 
         - steps (int): 采样步数（扩散时间步数量）。 
         - eta (float, optional): DDIM 的退火系数，范围 [0, 1]，仅在 DDIM 调度器中生效。默认 0.0。 
         - guidance_scale (float, optional): 无分类器引导强度。大于 1.0 时启用 CFG，引导强度越大越贴合条件。默认 7.5。 
         - use_reward (int, optional): 是否启用奖励引导（非 0 表示启用）。启用后将在每步对噪声预测加入梯度修正。默认 0。 
         - reward_target (Any, optional): 奖励引导的目标设定或上下文，具体含义由外部奖励模块定义。默认 None。 
         - target (Any, optional): 奖励相关的目标参数，占位或透传用途，具体含义依赖上层实现。默认 1。 
         - guidance (Any, optional): 奖励相关的引导系数或开关，占位或透传用途，具体含义依赖上层实现。默认 1。 generator (Optional[torch.Generator], optional): 随机数生成器，用于控制可复现性。默认 None。 
         - device (torch.device, optional): 计算设备（同时用于调度器时间步）。默认 "cuda:0"。 disable_prog (bool, optional): 是否禁用进度条显示。默认 True。

        Returns: Tuple[torch.FloatTensor, List[torch.FloatTensor], Optional[Any]]: - torch.FloatTensor: 最终采样得到的潜变量张量。 - List[torch.FloatTensor]: 每一步的中间潜变量（通常为上一时刻的样本）列表，长度等于 steps。 - Optional[Any]: 启用奖励引导时返回评估/奖励相关的输出 eva_out，否则为 None。

        Raises: AssertionError: 当 steps <= 0 时抛出。

        Notes: - 初始潜变量按标准正态分布采样，并由调度器的 init_noise_sigma 进行缩放。 - 当 guidance_scale > 1.0 时，采用无分类器引导：以全零条件作为无条件分支并按 scale 进行线性插值。 - 当 use_reward 启用时，将调用 compute_gradient 对噪声预测进行基于奖励的梯度修正（与调度器 alpha 序列相关）。 """
        assert steps > 0, f"{steps} must > 0."
        do_classifier_free_guidance = guidance_scale > 1.0
        # init latents
        bsz = cond.shape[0]
        if do_classifier_free_guidance:
            bsz = bsz // 2

        latents = torch.randn(
            (shape),
            generator=generator,
            device=cond.device,
            dtype=cond.dtype,
        )
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(steps, device=device)
        timesteps = self.scheduler.timesteps.to(device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        if eta != 0.0:
            extra_step_kwargs = {"eta": eta, "generator": generator}
        else:
            extra_step_kwargs = {"generator": generator}
        latent_list = []
        if do_classifier_free_guidance:
            un_cond = torch.zeros_like(cond).float()
            cond = torch.cat([un_cond, cond], dim=0)

        for i, t in enumerate(
            tqdm(timesteps, disable=disable_prog, desc="DDIM Sampling:", leave=False)
        ):
            # expand the latents if we are doing classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            # predict the noise residual
            timestep_tensor = torch.tensor([t], dtype=torch.long, device=device)
            timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
            noise_pred = diffusion_model.forward(
                latent_model_input, timestep_tensor, cond=cond
            )
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            if use_reward:
                # print('use_reward:', use_reward)
                sqrt_1minus_alpha_t = (1 - self.scheduler.alphas_cumprod[t]) ** 0.5
                computed_gradient, eva_out = self.compute_gradient(latents, step=i)
                noise_pred += (
                    (sqrt_1minus_alpha_t * self.target_guidance * computed_gradient)
                    .permute(1, 0, 2)
                    .contiguous()
                )
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
            latent_list.append(latents.prev_sample)  # pred_original_sample
            latents = latents.prev_sample
        # print(latents[0])
        if use_reward:
            return latents, latent_list, eva_out
        return latents, latent_list, None


class DiffTipeline_old(StableDiffusionPipeline):
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
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
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
        num_channels_latents = self.unet.config.in_channels
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

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

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
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                ############################################################
                ############################################################
                ## Guided Diffusion Modification ##

                ## grad = nabla_x 0.5 * || y - mu(x) ||^2
                ## nabla_x log p_t (y|x_t) = - [1/sigma^2] * grad

                ## For DDIM scheduler,
                ## modified noise = original noise - sqrt( 1-alpha_t ) * (nabla_x log p_t (y|x_t)) ,
                ## see eq(14) of http://arxiv.org/abs/2105.05233

                ## self.target_guidance <---> 1 / sigma^2
                ## self.target  <---> y

                target = torch.FloatTensor([[self.target]]).to(latents.device)
                target = target.repeat(batch_size * num_images_per_prompt, 1)
                sqrt_1minus_alpha_t = (1 - self.scheduler.alphas_cumprod[t]) ** 0.5
                noise_pred += (
                    sqrt_1minus_alpha_t
                    * self.target_guidance
                    * self.compute_gradient(latents, target=target)
                )

                ############################################################
                ############################################################

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            #############################################
            ## Disabled for correct evaluation of the reward
            #############################################
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            #############################################
            ## Disabled for correct evaluation of the reward
            #############################################
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        ##############
        has_nsfw_concept = False
        ##############

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    def setup_reward_model(self, reward_model):
        self.reward_model = reward_model
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()

    def set_target(self, target):
        self.target = target

    def set_guidance(self, guidance):
        self.target_guidance = guidance

    @torch.enable_grad()
    def compute_gradient(self, latent, target):
        latent.requires_grad_(True)
        out = self.reward_model(latent)
        l2_error = 0.5 * torch.nn.MSELoss()(out, target)
        self.reward_model.zero_grad()
        l2_error.backward()
        return latent.grad.clone()
