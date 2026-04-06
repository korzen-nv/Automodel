# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stable Diffusion 3.5 model processor for preprocessing.

Handles SD3.5 architecture models with:
- VAE for image encoding (16-channel latents)
- Dual CLIP text encoders (CLIP-L + CLIP-G, pooled dim 2048)
- T5 text encoder
"""

import logging
from typing import Any, Dict

import torch
from torch import autocast

from nemo_automodel.shared.transformers_patches import patch_t5_layer_norm

from .base import BaseModelProcessor
from .registry import ProcessorRegistry

logger = logging.getLogger(__name__)


@ProcessorRegistry.register("sd3")
class SD3Processor(BaseModelProcessor):
    """
    Processor for Stable Diffusion 3.5 (MMDiT) architecture models.

    SD3.5 uses a VAE for image encoding and three text encoders:
    - CLIP-L (text_encoder): pooled output 768-dim
    - CLIP-G (text_encoder_2): pooled output 1280-dim
    - T5-XXL (text_encoder_3): sequence embeddings

    The two CLIP pooled outputs are concatenated to produce
    pooled_projections of dim 2048 (768 + 1280).
    """

    @property
    def model_type(self) -> str:
        return "sd3"

    @property
    def default_model_name(self) -> str:
        return "stabilityai/stable-diffusion-3.5-large"

    def load_models(self, model_name: str, device: str) -> Dict[str, Any]:
        """
        Load SD3.5 models from StableDiffusion3Pipeline.

        Args:
            model_name: HuggingFace model path
            device: Device to load models on

        Returns:
            Dict containing VAE, tokenizers, and text encoders.
        """
        from diffusers import StableDiffusion3Pipeline

        logger.info("[SD3] Loading models from %s via StableDiffusion3Pipeline...", model_name)

        patch_t5_layer_norm()

        # Load pipeline without transformer (not needed for preprocessing)
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_name,
            transformer=None,
            torch_dtype=torch.bfloat16,
        )

        models = {}

        logger.info("  Configuring VAE...")
        models["vae"] = pipeline.vae.to(device=device, dtype=torch.bfloat16)
        models["vae"].eval()
        logger.debug("VAE config: %s", models["vae"].config)
        logger.debug("VAE scaling_factor: %s", models["vae"].config.scaling_factor)
        logger.debug("VAE shift_factor: %s", models["vae"].config.shift_factor)

        # CLIP-L (text_encoder)
        logger.info("  Configuring CLIP-L (text_encoder)...")
        models["clip_tokenizer"] = pipeline.tokenizer
        models["clip_encoder"] = pipeline.text_encoder.to(device)
        models["clip_encoder"].eval()

        # CLIP-G (text_encoder_2)
        logger.info("  Configuring CLIP-G (text_encoder_2)...")
        models["clip_tokenizer_2"] = pipeline.tokenizer_2
        models["clip_encoder_2"] = pipeline.text_encoder_2.to(device)
        models["clip_encoder_2"].eval()

        # T5-XXL (text_encoder_3)
        logger.info("  Configuring T5-XXL (text_encoder_3)...")
        models["t5_tokenizer"] = pipeline.tokenizer_3
        models["t5_encoder"] = pipeline.text_encoder_3.to(device)
        models["t5_encoder"].eval()

        del pipeline
        torch.cuda.empty_cache()

        logger.info("[SD3] Models loaded successfully!")
        return models

    def encode_image(
        self,
        image_tensor: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> torch.Tensor:
        """
        Encode image to latent space using SD3.5 VAE.

        Args:
            image_tensor: Image tensor (1, 3, H, W), normalized to [-1, 1]
            models: Dict containing 'vae'
            device: Device to use

        Returns:
            Latent tensor (C, H//8, W//8), FP16
        """
        vae = models["vae"]
        image_tensor = image_tensor.to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            latent = vae.encode(image_tensor).latent_dist.sample()

        # Apply SD3 scaling: (latent - shift_factor) * scaling_factor
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor

        return latent.detach().cpu().to(torch.float16).squeeze(0)

    def encode_text(
        self,
        prompt: str,
        models: Dict[str, Any],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text using dual CLIP + T5 encoders.

        Returns:
            Dict containing:
                - clip_tokens: CLIP-L token IDs
                - clip_hidden: CLIP-L hidden states
                - pooled_prompt_embeds: Concatenated CLIP-L + CLIP-G pooled (dim 2048)
                - t5_tokens: T5 token IDs
                - prompt_embeds: T5 hidden states
        """
        device_type = "cuda" if "cuda" in device else "cpu"

        # CLIP-L encoding
        clip_tokens = models["clip_tokenizer"](
            prompt,
            padding="max_length",
            max_length=models["clip_tokenizer"].model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        clip_output = models["clip_encoder"](
            clip_tokens.input_ids.to(device_type), output_hidden_states=True
        )
        clip_hidden = clip_output.hidden_states[-2]
        clip_pooled = clip_output.text_embeds  # CLIPTextModelWithProjection -> text_embeds

        # CLIP-G encoding
        clip_tokens_2 = models["clip_tokenizer_2"](
            prompt,
            padding="max_length",
            max_length=models["clip_tokenizer_2"].model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        clip_output_2 = models["clip_encoder_2"](
            clip_tokens_2.input_ids.to(device_type), output_hidden_states=True
        )
        clip_pooled_2 = clip_output_2.text_embeds  # CLIPTextModelWithProjection -> text_embeds

        # Concatenate CLIP-L (768) + CLIP-G (1280) pooled -> 2048
        pooled_prompt_embeds = torch.cat([clip_pooled, clip_pooled_2], dim=-1)

        # T5 encoding
        t5_tokens = models["t5_tokenizer"](
            prompt,
            padding="max_length",
            max_length=models["t5_tokenizer"].model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        t5_output = models["t5_encoder"](
            t5_tokens.input_ids.to(device_type), output_hidden_states=False
        )
        prompt_embeds = t5_output.last_hidden_state

        return {
            "clip_tokens": clip_tokens["input_ids"].cpu(),
            "clip_hidden": clip_hidden.detach().cpu().to(torch.bfloat16),
            "pooled_prompt_embeds": pooled_prompt_embeds.detach().cpu().to(torch.bfloat16),
            "t5_tokens": t5_tokens["input_ids"].cpu(),
            "prompt_embeds": prompt_embeds.detach().cpu().to(torch.bfloat16),
        }

    def verify_latent(
        self,
        latent: torch.Tensor,
        models: Dict[str, Any],
        device: str,
    ) -> bool:
        """
        Verify latent can be decoded back to reasonable image.
        """
        try:
            vae = models["vae"]
            device_type = "cuda" if "cuda" in device else "cpu"

            latent = latent.unsqueeze(0).to(device).float()

            with torch.no_grad(), autocast(device_type=device_type, dtype=torch.float32):
                # Undo SD3 scaling: latent / scaling_factor + shift_factor
                latent = latent / vae.config.scaling_factor + vae.config.shift_factor
                decoded = vae.decode(latent).sample

            _, c, h, w = decoded.shape
            if c != 3:
                return False

            if torch.isnan(decoded).any() or torch.isinf(decoded).any():
                return False

            return True

        except Exception as e:
            logger.warning("[SD3] Verification failed: %s", e)
            return False

    def get_cache_data(
        self,
        latent: torch.Tensor,
        text_encodings: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct cache dictionary for SD3.5.

        Uses same keys as FLUX for compatibility with TextToImageDataset
        and collate_fn_text_to_image.
        """
        return {
            # Image latent
            "latent": latent,
            # CLIP embeddings (CLIP-L hidden states for compatibility)
            "clip_tokens": text_encodings["clip_tokens"],
            "clip_hidden": text_encodings["clip_hidden"],
            # Concatenated CLIP-L + CLIP-G pooled (dim 2048)
            "pooled_prompt_embeds": text_encodings["pooled_prompt_embeds"],
            # T5 embeddings
            "t5_tokens": text_encodings["t5_tokens"],
            "prompt_embeds": text_encodings["prompt_embeds"],
            # Metadata
            "original_resolution": metadata["original_resolution"],
            "bucket_resolution": metadata["bucket_resolution"],
            "crop_offset": metadata["crop_offset"],
            "prompt": metadata["prompt"],
            "image_path": metadata["image_path"],
            "bucket_id": metadata["bucket_id"],
            "aspect_ratio": metadata["aspect_ratio"],
            # Model info
            "model_type": self.model_type,
        }
