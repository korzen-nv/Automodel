"""
FLUX.1-dev LoRA fine-tuning with DreamBooth-style trigger token.
Uses diffusers + peft directly, bypassing nemo-automodel.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FluxPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurgicalDataset(Dataset):
    """Load images + captions from atlas120k_flux directory."""

    def __init__(self, data_dir, resolution=512):
        self.data_dir = Path(data_dir)
        self.resolution = resolution

        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded {len(self.metadata)} samples from {data_dir}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        img_path = self.data_dir / entry["file_name"]
        caption = f"krnsurg {entry['text']}"

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)

        import torchvision.transforms as T

        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        pixel_values = transform(image)

        return {"pixel_values": pixel_values, "caption": caption}


def encode_prompt(text, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device, dtype):
    """Encode prompt using both CLIP and T5 encoders."""
    clip_inputs = tokenizer(
        text, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    ).to(device)
    clip_output = text_encoder(**clip_inputs, output_hidden_states=False)
    pooled_prompt_embeds = clip_output.pooler_output.to(dtype)

    t5_inputs = tokenizer_2(
        text, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
    ).to(device)
    prompt_embeds = text_encoder_2(**t5_inputs)[0].to(dtype)

    return prompt_embeds, pooled_prompt_embeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_every_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    args = parser.parse_args()

    project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="no",
        project_config=project_config,
    )

    set_seed(args.seed)
    dtype = torch.bfloat16

    logger.info("Loading FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=dtype)

    vae = pipe.vae
    transformer = pipe.transformer
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2

    del pipe
    torch.cuda.empty_cache()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    transformer.requires_grad_(False)

    vae.to(accelerator.device, dtype=dtype)
    text_encoder.to(accelerator.device, dtype=dtype)
    text_encoder_2.to(accelerator.device, dtype=dtype)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "proj_out",
        ],
        lora_dropout=0.0,
    )
    transformer.to(accelerator.device, dtype=dtype)
    transformer = get_peft_model(transformer, lora_config)
    for _, param in transformer.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype)
    transformer.print_trainable_parameters()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    dataset = SurgicalDataset(args.data_dir, resolution=args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)

    num_training_steps = args.num_epochs * math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(100, num_training_steps // 10),
        num_training_steps=num_training_steps,
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    logger.info(f"Training LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    logger.info(f"Epochs: {args.num_epochs}, Steps/epoch: {len(dataloader)}")
    logger.info(
        "Effective batch: "
        f"{args.train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes}"
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        transformer.train()
        epoch_loss = 0.0
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in progress:
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=dtype)
                captions = batch["caption"]

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                prompt_embeds_list = []
                pooled_embeds_list = []
                with torch.no_grad():
                    for cap in captions:
                        pe, ppe = encode_prompt(
                            cap,
                            tokenizer,
                            tokenizer_2,
                            text_encoder,
                            text_encoder_2,
                            accelerator.device,
                            dtype,
                        )
                        prompt_embeds_list.append(pe)
                        pooled_embeds_list.append(ppe)

                prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
                pooled_prompt_embeds = torch.cat(pooled_embeds_list, dim=0)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                sigmas = torch.sigmoid(torch.randn(bsz, device=latents.device, dtype=torch.float32))
                sigma = sigmas.view(-1, 1, 1, 1).to(dtype=latents.dtype)
                noisy_latents = sigma * noise + (1.0 - sigma) * latents
                target = noise - latents

                b, c, h, w = noisy_latents.shape
                packed = noisy_latents.view(b, c, h // 2, 2, w // 2, 2)
                packed = packed.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)

                img_ids = torch.zeros(h // 2, w // 2, 3, device=latents.device, dtype=dtype)
                img_ids[..., 1] = torch.arange(h // 2, device=latents.device)[:, None]
                img_ids[..., 2] = torch.arange(w // 2, device=latents.device)[None, :]
                img_ids = img_ids.reshape(-1, 3)

                txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=latents.device, dtype=dtype)

                guidance = None
                if transformer.config.guidance_embeds:
                    guidance = torch.ones(bsz, device=latents.device, dtype=torch.float32)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model_pred = transformer(
                        hidden_states=packed,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        timestep=sigmas,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance,
                        return_dict=False,
                    )[0]

                model_pred = model_pred.view(b, h // 2, w // 2, c, 2, 2)
                model_pred = model_pred.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.detach().item()
            global_step += 1

            if accelerator.is_main_process:
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

            if global_step % args.save_every_steps == 0 and accelerator.is_main_process:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                unwrapped = accelerator.unwrap_model(transformer)
                unwrapped.save_pretrained(save_dir)
                logger.info(f"Saved LoRA checkpoint to {save_dir}")

        avg_loss = epoch_loss / len(dataloader)
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1} avg_loss={avg_loss:.6f}")

    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "final")
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(save_dir)
        logger.info(f"Saved final LoRA to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
