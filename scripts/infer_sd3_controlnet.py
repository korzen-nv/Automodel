#!/usr/bin/env python3
"""
SD3.5 ControlNet inference — generate images conditioned on segmentation masks.

Usage:
    python scripts/infer_sd3_controlnet.py \
        --controlnet_path path/to/trained_controlnet \
        --manifest path/to/val_eval.jsonl \
        --output_dir path/to/output \
        --num_samples 10

    # With fine-tuned transformer:
    python scripts/infer_sd3_controlnet.py \
        --controlnet_path path/to/trained_controlnet \
        --transformer_path path/to/epoch_48/model/consolidated \
        --manifest path/to/val_eval.jsonl \
        --output_dir path/to/output
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SD3.5 ControlNet inference")
    parser.add_argument("--controlnet_path", type=str, required=True, help="Path to trained ControlNet checkpoint")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--transformer_path", type=str, default=None, help="Optional fine-tuned transformer path")
    parser.add_argument("--manifest", type=str, required=True, help="JSONL manifest with conditioning_image and text")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # Patch T5 layer norm for bf16
    if torch_dtype == torch.bfloat16:
        from nemo_automodel.shared.transformers_patches import patch_t5_layer_norm
        patch_t5_layer_norm()

    from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline

    logger.info("Loading ControlNet from %s", args.controlnet_path)
    controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch_dtype)

    logger.info("Loading pipeline from %s", args.model_id)
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.model_id,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )

    # Optionally load fine-tuned transformer
    if args.transformer_path:
        from diffusers import SD3Transformer2DModel
        logger.info("Loading fine-tuned transformer from %s", args.transformer_path)
        pipe.transformer = SD3Transformer2DModel.from_pretrained(
            args.transformer_path, torch_dtype=torch_dtype
        ).to("cuda")

    pipe.enable_model_cpu_offload()
    logger.info("Pipeline ready: %s", type(pipe).__name__)

    # Load manifest
    with open(args.manifest) as f:
        rows = [json.loads(line) for line in f]
    rows = rows[:args.num_samples]
    logger.info("Generating %d samples", len(rows))

    negative_prompt = "text, letters, watermark, words, writing, caption, label, subtitle, logo"

    for i, row in enumerate(rows):
        prompt = row["text"]
        control_image_path = row["conditioning_image"]
        control_image = Image.open(control_image_path).convert("RGB")

        logger.info("[%d/%d] Prompt: %s", i + 1, len(rows), prompt[:80])
        logger.info("  Control: %s", control_image_path)

        generator = torch.Generator(device="cuda").manual_seed(args.seed + i)

        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                control_image=control_image,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                negative_prompt=negative_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
            )

        image = output.images[0]
        safe_name = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt)[:50].strip().replace(" ", "_")
        image_path = output_dir / f"sample_{i:03d}_{safe_name}.png"
        image.save(image_path)

        # Also save the control image side-by-side for comparison
        control_resized = control_image.resize((args.width, args.height))
        control_path = output_dir / f"control_{i:03d}_{safe_name}.png"
        control_resized.save(control_path)

        logger.info("  Saved: %s", image_path)

    logger.info("Done — %d samples saved to %s", len(rows), output_dir)


if __name__ == "__main__":
    main()
