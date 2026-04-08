#!/usr/bin/env python3
"""
FLUX ControlNet inference on Atlas120k segmentation masks.

Usage:
    python scripts/infer_flux_controlnet.py \
        --controlnet_path path/to/checkpoint/flux_controlnet \
        --manifest path/to/val_eval.jsonl \
        --output_dir path/to/output \
        --num_samples 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

AUTOMODEL_ROOT = Path(__file__).resolve().parents[1]
if str(AUTOMODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOMODEL_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLUX ControlNet inference")
    parser.add_argument("--controlnet_path", type=str, required=True, help="Path to trained ControlNet checkpoint")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--manifest", type=str, required=True, help="JSONL manifest with conditioning_image and text")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--control_guidance_start", type=float, default=0.0)
    parser.add_argument("--control_guidance_end", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def sanitize_name(text: str, max_len: int = 64) -> str:
    safe = "".join(c if c.isalnum() or c in " _-" else "" for c in text)
    safe = safe.strip().replace(" ", "_")
    return safe[:max_len] or "sample"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    if torch_dtype == torch.bfloat16:
        try:
            from nemo_automodel.shared.transformers_patches import patch_t5_layer_norm
        except ImportError:
            logger.warning("nemo_automodel patch import unavailable; continuing without T5 layer-norm patch")
        else:
            patch_t5_layer_norm()

    from diffusers import FluxControlNetModel, FluxControlNetPipeline

    logger.info("Loading ControlNet from %s", args.controlnet_path)
    controlnet = FluxControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch_dtype)

    logger.info("Loading FLUX pipeline from %s", args.model_id)
    pipe = FluxControlNetPipeline.from_pretrained(
        args.model_id,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)
    logger.info("Pipeline ready: %s", type(pipe).__name__)

    with open(args.manifest) as handle:
        rows = [json.loads(line) for line in handle]
    rows = rows[: args.num_samples]
    logger.info("Generating %d samples from %s", len(rows), args.manifest)

    for i, row in enumerate(rows):
        prompt = row["text"]
        control_image_path = row["conditioning_image"]
        control_image = Image.open(control_image_path).convert("RGB")

        logger.info("[%d/%d] Prompt: %s", i + 1, len(rows), prompt[:120])
        logger.info("  Control: %s", control_image_path)

        generator = torch.Generator(device="cuda").manual_seed(args.seed + i)
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                control_image=control_image,
                control_guidance_start=args.control_guidance_start,
                control_guidance_end=args.control_guidance_end,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
            )

        image = output.images[0]
        frame_name = row.get("frame_name", f"sample_{i:03d}")
        prompt_stub = sanitize_name(prompt)

        image_path = output_dir / f"sample_{i:03d}_{frame_name}_{prompt_stub}.png"
        image.save(image_path)

        control_path = output_dir / f"control_{i:03d}_{frame_name}.png"
        control_image.resize((args.width, args.height)).save(control_path)

        logger.info("  Saved: %s", image_path)

    logger.info("Done — %d samples saved to %s", len(rows), output_dir)


if __name__ == "__main__":
    main()
