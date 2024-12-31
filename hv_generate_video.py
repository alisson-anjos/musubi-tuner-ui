import argparse
from datetime import datetime
from pathlib import Path
import random
import sys
import os
import time
from typing import Optional, Union

import numpy as np
import torch
import torchvision
import accelerate
from diffusers.utils.torch_utils import randn_tensor
from transformers.models.llama import LlamaModel
from tqdm import tqdm
import av
from einops import rearrange
from safetensors.torch import load_file

from hunyuan_model import vae
from hunyuan_model.text_encoder import TextEncoder
from hunyuan_model.text_encoder import PROMPT_TEMPLATE
from hunyuan_model.vae import load_vae
from hunyuan_model.models import load_transformer, get_rotary_pos_embed
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from networks import lora

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def clean_memory_on_device(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "cpu":
        pass
    elif device.type == "mps":  # not tested
        torch.mps.empty_cache()


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # # save video with av
    # container = av.open(path, "w")
    # stream = container.add_stream("libx264", rate=fps)
    # for x in outputs:
    #     frame = av.VideoFrame.from_ndarray(x, format="rgb24")
    #     packet = stream.encode(frame)
    #     container.mux(packet)
    # packet = stream.encode(None)
    # container.mux(packet)
    # container.close()

    height, width, _ = outputs[0].shape

    # create output container
    container = av.open(path, mode="w")

    # create video stream
    codec = "libx264"
    pixel_format = "yuv420p"
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pixel_format
    stream.bit_rate = 4000000  # 4Mbit/s

    for frame_array in outputs:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        packets = stream.encode(frame)
        for packet in packets:
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


# region Encoding prompt


def encode_prompt(prompt: Union[str, list[str]], device: torch.device, num_videos_per_prompt: int, text_encoder: TextEncoder):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_videos_per_prompt (`int`):
            number of videos that should be generated per prompt
        text_encoder (TextEncoder):
            text encoder to be used for encoding the prompt
    """
    # LoRA and Textual Inversion are not supported in this script
    # negative prompt and prompt embedding are not supported in this script
    # clip_skip is not supported in this script because it is not used in the original script
    data_type = "video"  # video only, image is not supported

    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

    with torch.no_grad():
        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, device=device)
    prompt_embeds = prompt_outputs.hidden_state

    attention_mask = prompt_outputs.attention_mask
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        bs_embed, seq_len = attention_mask.shape
        attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
        attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

    prompt_embeds_dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    if prompt_embeds.ndim == 2:
        bs_embed, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
    else:
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds, attention_mask


def encode_input_prompt(prompt, args, device, fp8_llm=False, accelerator=None):
    # constants
    prompt_template_video = "dit-llm-encode-video"
    prompt_template = "dit-llm-encode"
    text_encoder_dtype = torch.float16
    text_encoder_type = "llm"
    text_len = 256
    hidden_state_skip_layer = 2
    apply_final_norm = False
    reproduce = False

    text_encoder_2_type = "clipL"
    text_len_2 = 77

    num_videos = 1

    # if args.prompt_template_video is not None:
    #     crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
    # elif args.prompt_template is not None:
    #     crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
    # else:
    #     crop_start = 0
    crop_start = PROMPT_TEMPLATE[prompt_template_video].get("crop_start", 0)
    max_length = text_len + crop_start

    # prompt_template
    prompt_template = PROMPT_TEMPLATE[prompt_template]

    # prompt_template_video
    prompt_template_video = PROMPT_TEMPLATE[prompt_template_video]  # if args.prompt_template_video is not None else None

    # load text encoders
    logger.info(f"loading text encoder: {args.text_encoder1}")
    text_encoder = TextEncoder(
        text_encoder_type=text_encoder_type,
        max_length=max_length,
        text_encoder_dtype=text_encoder_dtype,
        text_encoder_path=args.text_encoder1,
        tokenizer_type=text_encoder_type,
        prompt_template=prompt_template,
        prompt_template_video=prompt_template_video,
        hidden_state_skip_layer=hidden_state_skip_layer,
        apply_final_norm=apply_final_norm,
        reproduce=reproduce,
    )
    text_encoder.eval()
    if fp8_llm:
        org_dtype = text_encoder.dtype
        logger.info(f"Moving and casting text encoder to {device} and torch.float8_e4m3fn")
        text_encoder.to(device=device, dtype=torch.float8_e4m3fn)

        # prepare LLM for fp8
        def prepare_fp8(llama_model: LlamaModel, target_dtype):
            def forward_hook(module):
                def forward(hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
                    return module.weight.to(input_dtype) * hidden_states.to(input_dtype)

                return forward

            for module in llama_model.modules():
                if module.__class__.__name__ in ["Embedding"]:
                    # print("set", module.__class__.__name__, "to", target_dtype)
                    module.to(target_dtype)
                if module.__class__.__name__ in ["LlamaRMSNorm"]:
                    # print("set", module.__class__.__name__, "hooks")
                    module.forward = forward_hook(module)

        prepare_fp8(text_encoder.model, org_dtype)

    logger.info(f"loading text encoder 2: {args.text_encoder2}")
    text_encoder_2 = TextEncoder(
        text_encoder_type=text_encoder_2_type,
        max_length=text_len_2,
        text_encoder_dtype=text_encoder_dtype,
        text_encoder_path=args.text_encoder2,
        tokenizer_type=text_encoder_2_type,
        reproduce=reproduce,
    )
    text_encoder_2.eval()

    # encode prompt
    logger.info(f"Encoding prompt with text encoder 1")
    text_encoder.to(device=device)
    if fp8_llm:
        with accelerator.autocast():
            prompt_embeds, prompt_mask = encode_prompt(prompt, device, num_videos, text_encoder)
    else:
        prompt_embeds, prompt_mask = encode_prompt(prompt, device, num_videos, text_encoder)
    text_encoder = None
    clean_memory_on_device(device)

    logger.info(f"Encoding prompt with text encoder 2")
    text_encoder_2.to(device=device)
    prompt_embeds_2, prompt_mask_2 = encode_prompt(prompt, device, num_videos, text_encoder_2)

    prompt_embeds = prompt_embeds.to("cpu")
    prompt_mask = prompt_mask.to("cpu")
    prompt_embeds_2 = prompt_embeds_2.to("cpu")
    prompt_mask_2 = prompt_mask_2.to("cpu")

    text_encoder_2 = None
    clean_memory_on_device(device)

    return prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2


# endregion


def decode_latents(args, latents, device):
    vae_dtype = torch.float16
    vae, _, s_ratio, t_ratio = load_vae(vae_dtype=vae_dtype, device=device, vae_path=args.vae)
    vae.eval()
    # vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

    # set chunk_size to CausalConv3d recursively
    chunk_size = args.vae_chunk_size
    if chunk_size is not None:
        vae.set_chunk_size_for_causal_conv_3d(chunk_size)
        logger.info(f"Set chunk_size to {chunk_size} for CausalConv3d")

    expand_temporal_dim = False
    if len(latents.shape) == 4:
        latents = latents.unsqueeze(2)
        expand_temporal_dim = True
    elif len(latents.shape) == 5:
        pass
    else:
        raise ValueError(f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}.")

    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    else:
        latents = latents / vae.config.scaling_factor

    latents = latents.to(device=device, dtype=vae.dtype)
    if args.vae_spatial_tile_sample_min_size is not None:
        vae.enable_spatial_tiling(True)
        vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
        vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
    # elif args.vae_tiling:
    else:
        vae.enable_spatial_tiling(True)
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]

    if expand_temporal_dim or image.shape[2] == 1:
        image = image.squeeze(2)

    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().float()

    return image


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo inference script")

    parser.add_argument("--dit", type=str, required=True, help="DiT checkpoint path or directory")
    parser.add_argument("--vae", type=str, required=True, help="VAE checkpoint path or directory")
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 directory")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 directory")

    # LoRA
    parser.add_argument("--lora_weight", type=str, required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, default=1.0, help="LoRA multiplier")

    parser.add_argument("--prompt", type=str, required=True, help="prompt for generation")
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size")
    parser.add_argument("--video_length", type=int, default=129, help="video length")
    parser.add_argument("--infer_steps", type=int, default=50, help="number of inference steps")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument("--embedded_cfg_scale", type=float, default=6.0, help="Embeded classifier free guidance scale.")

    # Flow Matching
    parser.add_argument("--flow_shift", type=float, default=7.0, help="Shift factor for flow matching schedulers.")

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for Text Encoder 1 (LLM)")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode", type=str, default="torch", choices=["flash", "torch", "sageattn", "sdpa"], help="attention mode"
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    parser.add_argument("--blocks_to_swap", type=int, default=None, help="number of blocks to swap in the model")
    parser.add_argument("--img_in_txt_in_offloading", action="store_true", help="offload img_in and txt_in to cpu")
    parser.add_argument("--output_type", type=str, default="video", help="output type: video, latent or both")
    parser.add_argument("--latent_path", type=str, default=None, help="path to latent for decode. no inference")

    args = parser.parse_args()

    assert args.latent_path is None or args.output_type == "video", "latent-path is only supported with output-type=video"

    # update dit_weight based on model_base if not exists

    return args


def check_inputs(args):
    height = args.video_size[0]
    width = args.video_size[1]
    video_length = args.video_length

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    return height, width, video_length


def main():
    args = parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dit_dtype = torch.bfloat16
    dit_weight_dtype = torch.float8_e4m3fn if args.fp8 else dit_dtype
    logger.info(f"Using device: {device}, DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}")

    if args.latent_path is not None:
        latents = torch.load(args.latent_path, map_location="cpu")
        logger.info(f"Loaded latent from {args.latent_path}. Shape: {latents.shape}")
        latents = latents.unsqueeze(0)
        seeds = [0]  # dummy seed
    else:
        # prepare accelerator
        mixed_precision = "bf16" if dit_dtype == torch.bfloat16 else "fp16"
        accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)

        # load prompt
        prompt = args.prompt  # TODO load prompts from file
        assert prompt is not None, "prompt is required"

        # check inputs: may be height, width, video_length etc will be changed for each generation in future
        height, width, video_length = check_inputs(args)

        # encode prompt with LLM and Text Encoder
        logger.info(f"Encoding prompt: {prompt}")
        prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2 = encode_input_prompt(
            prompt, args, device, args.fp8_llm, accelerator
        )

        # load DiT model
        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
        loading_device = "cpu" if blocks_to_swap > 0 else device

        logger.info(f"Loading DiT model from {args.dit}")
        if args.attn_mode == "sdpa":
            args.attn_mode = "torch"
        transformer = load_transformer(args.dit, args.attn_mode, loading_device, dit_dtype)
        transformer.eval()

        # load LoRA weights
        if args.lora_weight is not None:
            logger.info(f"Loading LoRA weights from {args.lora_weight}")
            weights_sd = load_file(args.lora_weight)
            network = lora.create_network_from_weights_hunyuan_video(
                args.lora_multiplier, weights_sd, unet=transformer, for_inference=True
            )
            logger.info("Merging LoRA weights to DiT model")
            network.merge_to(None, transformer, weights_sd, device=device)
            logger.info("LoRA weights loaded")

        if blocks_to_swap > 0:
            logger.info(f"Casting model to {dit_weight_dtype}")
            transformer.to(dtype=dit_weight_dtype)
            logger.info(f"Enable swap {blocks_to_swap} blocks to CPU from device: {device}")
            transformer.enable_block_swap(blocks_to_swap, device, supports_backward=False)
            transformer.move_to_device_except_swap_blocks(device)
            transformer.prepare_block_swap_before_forward()
        else:
            logger.info(f"Moving and casting model to {device} and {dit_weight_dtype}")
            transformer.to(device=device, dtype=dit_weight_dtype)
        if args.img_in_txt_in_offloading:
            logger.info("Enable offloading img_in and txt_in to CPU")
            transformer.enable_img_in_txt_in_offloading()

        # load scheduler
        logger.info(f"Loading scheduler")
        scheduler = FlowMatchDiscreteScheduler(shift=args.flow_shift, reverse=True, solver="euler")

        # Prepare timesteps
        num_inference_steps = args.infer_steps
        scheduler.set_timesteps(num_inference_steps, device=device)  # n_tokens is not used in FlowMatchDiscreteScheduler
        timesteps = scheduler.timesteps

        # Prepare generator
        num_videos_per_prompt = 1  # args.num_videos
        seed = args.seed
        if seed is None:
            seeds = [random.randint(0, 1_000_000) for _ in range(num_videos_per_prompt)]
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(num_videos_per_prompt)]
        else:
            raise ValueError(f"Seed must be an integer or None, got {seed}.")
        generator = [torch.Generator(device).manual_seed(seed) for seed in seeds]

        # Prepare latents
        num_channels_latents = 16  # transformer.config.in_channels
        vae_scale_factor = 2 ** (4 - 1)  # len(self.vae.config.block_out_channels) == 4

        vae_ver = vae.VAE_VER
        if "884" in vae_ver:
            latent_video_length = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            latent_video_length = (video_length - 1) // 8 + 1
        else:
            latent_video_length = video_length

        shape = (
            num_videos_per_prompt,
            num_channels_latents,
            latent_video_length,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dit_dtype)
        # FlowMatchDiscreteScheduler does not have init_noise_sigma

        # Denoising loop
        embedded_guidance_scale = args.embedded_cfg_scale
        if embedded_guidance_scale is not None:
            guidance_expand = torch.tensor([embedded_guidance_scale * 1000.0] * latents.shape[0], dtype=torch.float32, device="cpu")
            guidance_expand = guidance_expand.to(device=device, dtype=dit_dtype)
        else:
            guidance_expand = None
        freqs_cos, freqs_sin = get_rotary_pos_embed(vae.VAE_VER, transformer, video_length, height, width)
        # n_tokens = freqs_cos.shape[0]

        # move and cast all inputs to the correct device and dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=dit_dtype)
        prompt_mask = prompt_mask.to(device=device)
        prompt_embeds_2 = prompt_embeds_2.to(device=device, dtype=dit_dtype)
        prompt_mask_2 = prompt_mask_2.to(device=device)
        freqs_cos = freqs_cos.to(device=device, dtype=dit_dtype)
        freqs_sin = freqs_sin.to(device=device, dtype=dit_dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents = scheduler.scale_model_input(latents, t)

                # predict the noise residual
                with torch.no_grad(), accelerator.autocast():
                    noise_pred = transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                        latents,  # [1, 16, 33, 24, 42]
                        t.repeat(latents.shape[0]).to(device=device, dtype=dit_dtype),  # [1]
                        text_states=prompt_embeds,  # [1, 256, 4096]
                        text_mask=prompt_mask,  # [1, 256]
                        text_states_2=prompt_embeds_2,  # [1, 768]
                        freqs_cos=freqs_cos,  # [seqlen, head_dim]
                        freqs_sin=freqs_sin,  # [seqlen, head_dim]
                        guidance=guidance_expand,
                        return_dict=True,
                    )["x"]

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()

        # print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

        latents = latents.detach().cpu()
        transformer = None
        clean_memory_on_device(device)

    # Save samples
    output_type = args.output_type
    save_path = args.save_path  # if args.save_path_suffix == "" else f"{args.save_path}_{args.save_path_suffix}"
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    if output_type == "latent" or output_type == "both":
        # save latent
        for i, latent in enumerate(latents):
            latent_path = f"{save_path}/{time_flag}_{i}_{seeds[i]}_latent.pt"
            torch.save(latent, latent_path)
            logger.info(f"Latent save to: {latent_path}")
    if output_type == "video" or output_type == "both":
        # save video
        videos = decode_latents(args, latents, device)
        for i, sample in enumerate(videos):
            sample = sample.unsqueeze(0)
            save_path = f"{save_path}/{time_flag}_{seeds[i]}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f"Sample save to: {save_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()