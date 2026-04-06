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
Stable Diffusion 3.5 model adapter for FlowMatching Pipeline.

This adapter supports SD3.5 (MMDiT architecture) models with:
- T5 text embeddings (encoder_hidden_states)
- Concatenated CLIP-L + CLIP-G pooled embeddings (pooled_projections, dim 2048)
- 4D image latents passed directly (patchified internally by the model)
"""

import random
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import FlowMatchingContext, ModelAdapter


class SD3Adapter(ModelAdapter):
    """
    Model adapter for Stable Diffusion 3.5 (MMDiT) image generation models.

    Supports batch format from multiresolution dataloader:
    - image_latents: [B, C, H, W] (16-channel VAE latents)
    - text_embeddings: T5 embeddings [B, seq_len, dim]
    - pooled_prompt_embeds: Concatenated CLIP-L + CLIP-G pooled [B, 2048]

    SD3Transformer2DModel forward interface:
    - hidden_states: [B, C, H, W] raw latents (patchified internally)
    - encoder_hidden_states: T5 text embeddings
    - pooled_projections: Concatenated CLIP pooled embeddings
    - timestep: Timesteps in [0, num_train_timesteps] range
    """

    def __init__(self, guidance_scale: float = 7.0):
        """
        Initialize SD3Adapter.

        Args:
            guidance_scale: Guidance scale for classifier-free guidance
                (used during inference; during training, CFG dropout is handled
                by the pipeline's cfg_dropout_prob).
        """
        self.guidance_scale = guidance_scale

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        """
        Prepare inputs for SD3Transformer2DModel from FlowMatchingContext.

        Key differences from FluxAdapter:
        - No latent packing: SD3 handles patchification internally via PatchEmbed
        - No positional IDs: SD3 uses learned positional embeddings internally
        - Timesteps are NOT divided by 1000 (passed as-is from the pipeline)
        - pooled_projections dimension is 2048 (concatenated dual CLIP)
        """
        batch = context.batch
        device = context.device
        dtype = context.dtype

        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 4:
            raise ValueError(f"SD3Adapter expects 4D latents [B, C, H, W], got {noisy_latents.ndim}D")

        batch_size = noisy_latents.shape[0]

        # Get text embeddings (T5)
        text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)

        # Get pooled embeddings (concatenated CLIP-L + CLIP-G, dim 2048)
        if "pooled_prompt_embeds" in batch:
            pooled_projections = batch["pooled_prompt_embeds"].to(device, dtype=dtype)
        else:
            pooled_projections = torch.zeros(batch_size, 2048, device=device, dtype=dtype)

        if pooled_projections.ndim == 1:
            pooled_projections = pooled_projections.unsqueeze(0)

        # CFG dropout: zero out text conditioning with probability cfg_dropout_prob
        if random.random() < context.cfg_dropout_prob:
            text_embeddings = torch.zeros_like(text_embeddings)
            pooled_projections = torch.zeros_like(pooled_projections)

        # SD3 timesteps: pass as-is (not normalized to [0, 1] like FLUX)
        timesteps = context.timesteps.to(dtype)

        inputs = {
            "hidden_states": noisy_latents,
            "encoder_hidden_states": text_embeddings,
            "pooled_projections": pooled_projections,
            "timestep": timesteps,
            "return_dict": False,
        }

        return inputs

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute forward pass for SD3Transformer2DModel.

        Returns prediction in [B, C, H, W] format (no unpacking needed).
        """
        model_pred = model(**inputs)

        # Handle tuple output from return_dict=False
        pred = self.post_process_prediction(model_pred)

        return pred
