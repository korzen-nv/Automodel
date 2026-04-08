#!/usr/bin/env python3
"""Train FLUX ControlNet with one-hot segmentation conditioning on a single GPU."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

from accelerate.utils import set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.models.controlnets.controlnet import ControlNetConditioningEmbedding
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling


@dataclass
class PromptCacheItem:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FLUX ControlNet with one-hot conditioning")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--class_ids_path", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=12000)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--checkpoints_total_limit", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=200)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_double_layers", type=int, default=4)
    parser.add_argument("--num_single_layers", type=int, default=0)
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--conditioning_embedding_channels", type=int, default=64)
    parser.add_argument("--conditioning_embedding_out_channels", type=str, default="16,32,96,256")
    parser.add_argument("--image_interpolation_mode", type=str, default="lanczos")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", choices=["no", "bf16", "fp16"], default="bf16")
    return parser.parse_args()


class SegOneHotDataset(Dataset):
    def __init__(self, rows: list[dict], resolution: int, image_interpolation: InterpolationMode):
        self.rows = rows
        self.resolution = resolution
        self.image_interpolation = image_interpolation

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]

        image = Image.open(row["image"]).convert("RGB")
        image = TF.resize(image, self.resolution, interpolation=self.image_interpolation)
        image = TF.center_crop(image, [self.resolution, self.resolution])
        pixel_values = TF.to_tensor(image)
        pixel_values = TF.normalize(pixel_values, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        conditioning = torch.from_numpy(np.load(row["conditioning_image"], allow_pickle=False)).float()
        conditioning = TF.resize(conditioning, self.resolution, interpolation=InterpolationMode.NEAREST_EXACT)
        conditioning = TF.center_crop(conditioning, [self.resolution, self.resolution])

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning,
            "text": row["text"],
        }


def collate_fn(examples: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "conditioning_pixel_values": torch.stack([example["conditioning_pixel_values"] for example in examples]),
        "texts": [example["text"] for example in examples],
    }


def parse_block_channels(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def resolve_resume_checkpoint(output_dir: Path, resume_from_checkpoint: str | None) -> Path | None:
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint != "latest":
        checkpoint = Path(resume_from_checkpoint)
        return checkpoint if checkpoint.is_dir() else None

    checkpoints = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def cleanup_old_checkpoints(output_dir: Path, limit: int) -> None:
    if limit is None or limit <= 0:
        return
    checkpoints = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if len(checkpoints) < limit:
        return
    remove_count = len(checkpoints) - limit + 1
    for path in checkpoints[:remove_count]:
        print(f"Removing checkpoint: {path.name}", flush=True)
        shutil.rmtree(path)


def get_sigmas(
    timesteps: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    device: torch.device,
    dtype: torch.dtype,
    n_dim: int,
) -> torch.Tensor:
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def save_checkpoint(
    save_path: Path,
    flux_controlnet: FluxControlNetModel,
    conditioning_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    global_step: int,
    class_ids: list[int],
    conditioning_embedding_channels: int,
    conditioning_embedding_out_channels: tuple[int, ...],
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    flux_controlnet_state = {
        name: tensor.detach().cpu()
        for name, tensor in flux_controlnet.state_dict().items()
        if not name.startswith("input_hint_block.")
    }
    save_file(flux_controlnet_state, str(save_path / "flux_controlnet.safetensors"))
    (save_path / "flux_controlnet_config.json").write_text(
        json.dumps(dict(flux_controlnet.config), indent=2),
        encoding="utf-8",
    )
    save_file(conditioning_encoder.state_dict(), str(save_path / "conditioning_encoder.safetensors"))
    (save_path / "conditioning_encoder_config.json").write_text(
        json.dumps(
            {
                "class_ids": class_ids,
                "conditioning_embedding_channels": conditioning_embedding_channels,
                "conditioning_embedding_out_channels": list(conditioning_embedding_out_channels),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        save_path / "training_state.pt",
    )


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def load_checkpoint(
    checkpoint_path: Path,
    flux_controlnet: FluxControlNetModel,
    conditioning_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    device: torch.device,
) -> int:
    flux_controlnet_state = load_file(str(checkpoint_path / "flux_controlnet.safetensors"), device="cpu")
    missing_keys, unexpected_keys = flux_controlnet.load_state_dict(flux_controlnet_state, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected Flux ControlNet keys in {checkpoint_path}: {unexpected_keys}")
    if any(not key.startswith("input_hint_block.") for key in missing_keys):
        raise RuntimeError(f"Missing Flux ControlNet keys in {checkpoint_path}: {missing_keys}")

    conditioning_encoder.load_state_dict(load_file(str(checkpoint_path / "conditioning_encoder.safetensors")))

    state = torch.load(checkpoint_path / "training_state.pt", map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    move_optimizer_state_to_device(optimizer, device)
    return int(state["global_step"])


def build_prompt_cache(
    rows: list[dict],
    tokenizer_one,
    tokenizer_two,
    text_encoder_one,
    text_encoder_two,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> dict[str, PromptCacheItem]:
    unique_prompts = sorted({row["text"] for row in rows})
    prompt_cache: dict[str, PromptCacheItem] = {}

    pipe = FluxControlNetPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
        vae=None,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        transformer=None,
        controlnet=None,
    )
    pipe = pipe.to(device)
    for prompt in unique_prompts:
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(prompt, prompt_2=prompt)
        prompt_cache[prompt] = PromptCacheItem(
            prompt_embeds=prompt_embeds.to(dtype=weight_dtype).cpu(),
            pooled_prompt_embeds=pooled_prompt_embeds.to(dtype=weight_dtype).cpu(),
            text_ids=text_ids.to(dtype=weight_dtype).cpu(),
        )
    del pipe
    return prompt_cache


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir.resolve() if args.cache_dir is not None else None
    class_ids_path = args.class_ids_path or args.manifest.resolve().parent.parent / "class_ids.json"
    class_ids = json.loads(class_ids_path.read_text(encoding="utf-8"))
    conditioning_channels = len(class_ids)
    conditioning_embedding_out_channels = parse_block_channels(args.conditioning_embedding_out_channels)

    with args.manifest.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if not rows:
        raise RuntimeError(f"No rows found in {args.manifest}")

    image_interpolation = getattr(InterpolationMode, args.image_interpolation_mode.upper(), None)
    if image_interpolation is None:
        raise ValueError(f"Unsupported interpolation mode: {args.image_interpolation_mode}")

    device = torch.device("cuda")
    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[args.mixed_precision]

    print(f"Loading tokenizers/text encoders from {args.pretrained_model_name_or_path}", flush=True)
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", cache_dir=cache_dir
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", cache_dir=cache_dir
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", cache_dir=cache_dir
    ).to(device)
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", cache_dir=cache_dir
    ).to(device)

    prompt_cache = build_prompt_cache(
        rows,
        tokenizer_one,
        tokenizer_two,
        text_encoder_one,
        text_encoder_two,
        device=device,
        weight_dtype=weight_dtype,
    )

    text_encoder_one.to("cpu")
    text_encoder_two.to("cpu")
    del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
    torch.cuda.empty_cache()

    print("Loading FLUX components", flush=True)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", cache_dir=cache_dir
    ).to(device=device, dtype=weight_dtype)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", cache_dir=cache_dir
    ).to(device=device, dtype=weight_dtype)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=cache_dir
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae.requires_grad_(False)
    flux_transformer.requires_grad_(False)
    vae.eval()
    flux_transformer.eval()
    flux_controlnet = FluxControlNetModel.from_transformer(
        flux_transformer,
        num_layers=args.num_double_layers,
        num_single_layers=args.num_single_layers,
        attention_head_dim=flux_transformer.config["attention_head_dim"],
        num_attention_heads=flux_transformer.config["num_attention_heads"],
    ).to(device)
    conditioning_encoder = ControlNetConditioningEmbedding(
        conditioning_embedding_channels=args.conditioning_embedding_channels,
        conditioning_channels=conditioning_channels,
        block_out_channels=conditioning_embedding_out_channels,
    ).to(device)
    flux_controlnet.input_hint_block = conditioning_encoder
    flux_controlnet.train()
    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()
        flux_controlnet.enable_gradient_checkpointing()

    dataset = SegOneHotDataset(rows, args.resolution, image_interpolation)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    params = list(flux_controlnet.parameters())
    optimizer = AdamW(params, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    resume_checkpoint = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    global_step = 0
    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}", flush=True)
        global_step = load_checkpoint(
            resume_checkpoint,
            flux_controlnet,
            conditioning_encoder,
            optimizer,
            lr_scheduler,
            device=device,
        )

    progress_bar = tqdm(total=args.max_train_steps, initial=global_step, desc="Steps")

    def get_batch_prompt_tensors(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_embeds = torch.cat([prompt_cache[text].prompt_embeds for text in texts], dim=0).to(device)
        pooled_prompt_embeds = torch.cat([prompt_cache[text].pooled_prompt_embeds for text in texts], dim=0).to(device)
        text_ids = prompt_cache[texts[0]].text_ids.to(device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    step_in_epoch = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            step_in_epoch += 1
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            conditioning_values = batch["conditioning_pixel_values"].to(device=device, dtype=weight_dtype)
            prompt_embeds, pooled_prompt_embeds, text_ids = get_batch_prompt_tensors(batch["texts"])

            with torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=weight_dtype != torch.float32):
                pixel_latents_tmp = vae.encode(pixel_values).latent_dist.sample()
                pixel_latents_tmp = (pixel_latents_tmp - vae.config.shift_factor) * vae.config.scaling_factor
                pixel_latents = FluxControlNetPipeline._pack_latents(
                    pixel_latents_tmp,
                    pixel_values.shape[0],
                    pixel_latents_tmp.shape[1],
                    pixel_latents_tmp.shape[2],
                    pixel_latents_tmp.shape[3],
                )

                latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
                    batch_size=pixel_latents_tmp.shape[0],
                    height=pixel_latents_tmp.shape[2] // 2,
                    width=pixel_latents_tmp.shape[3] // 2,
                    device=device,
                    dtype=pixel_values.dtype,
                )

                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=device, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=device)
                sigmas = get_sigmas(
                    timesteps,
                    scheduler=noise_scheduler_copy,
                    device=device,
                    dtype=pixel_latents.dtype,
                    n_dim=pixel_latents.ndim,
                )
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                guidance_vec = (
                    torch.full((bsz,), args.guidance_scale, device=device, dtype=weight_dtype)
                    if flux_transformer.config.guidance_embeds
                    else None
                )

                controlnet_block_samples, controlnet_single_block_samples = flux_controlnet(
                    hidden_states=noisy_model_input,
                    controlnet_cond=conditioning_values,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )

                noise_pred = flux_transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_block_samples=[sample.to(dtype=weight_dtype) for sample in controlnet_block_samples],
                    controlnet_single_block_samples=[
                        sample.to(dtype=weight_dtype) for sample in controlnet_single_block_samples
                    ],
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(noise_pred.float(), (noise - pixel_latents).float(), reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=float(loss.detach()), lr=lr_scheduler.get_last_lr()[0])

            if global_step % args.checkpointing_steps == 0:
                cleanup_old_checkpoints(output_dir, args.checkpoints_total_limit)
                save_path = output_dir / f"checkpoint-{global_step}"
                save_checkpoint(
                    save_path=save_path,
                    flux_controlnet=flux_controlnet,
                    conditioning_encoder=conditioning_encoder,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    global_step=global_step,
                    class_ids=class_ids,
                    conditioning_embedding_channels=args.conditioning_embedding_channels,
                    conditioning_embedding_out_channels=conditioning_embedding_out_channels,
                )
                print(f"Saved checkpoint: {save_path}", flush=True)

            if global_step >= args.max_train_steps:
                break

    final_dir = output_dir / "final"
    save_checkpoint(
        save_path=final_dir,
        flux_controlnet=flux_controlnet,
        conditioning_encoder=conditioning_encoder,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        global_step=global_step,
        class_ids=class_ids,
        conditioning_embedding_channels=args.conditioning_embedding_channels,
        conditioning_embedding_out_channels=conditioning_embedding_out_channels,
    )
    print(f"Training complete at step {global_step}", flush=True)


if __name__ == "__main__":
    main()
