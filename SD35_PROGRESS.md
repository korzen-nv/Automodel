# SD3.5-Large Fine-Tuning on Atlas120k Surgical Data

## Summary

Fine-tuned Stable Diffusion 3.5-Large (~8B params, MMDiT architecture) on 8,574 surgical video frames from the Atlas120k dataset using NeMo AutoModel's flow matching pipeline. The model successfully generates realistic endoscopic/laparoscopic surgical images from text prompts.

## Key Results

- **Base SD3.5 (no fine-tuning)**: Produces generic OR photos, medical illustrations, and anatomical diagrams — nothing resembling real endoscopic footage. The trigger word "krnsurg" is unknown to the base model.
- **After fine-tuning (~50 epochs)**: Generates realistic laparoscopic surgical scenes with correct tissue textures, instrument types, camera angles, and lighting consistent with the atlas120k training data.
- **Convergence**: The biggest quality gain happens in the first ~10 epochs. After epoch 20, improvements plateau. Loss: ~0.24 (epoch 1) → ~0.22 (epoch 20) → ~0.22 (epoch 48).
- **Text rendering artifact**: Using a bare trigger word ("krnsurg") as the only caption caused SD3.5 to render it as visible text in images (SD3.5's text rendering capability). Fixed by using descriptive captions + negative prompt.

## Model

- **Base model**: `stabilityai/stable-diffusion-3.5-large`
- **Architecture**: MMDiT (Multimodal Diffusion Transformer), flow matching
- **Framework**: NeMo AutoModel with custom `SD3Adapter` + `SD3Processor`
- **Hardware**: 8x A100-SXM4-80GB (Draco OCI cluster)

## Dataset

- **Source**: Atlas120k surgical video frames (8,574 images, 14 surgical procedure types)
- **Caption**: `"krnsurg A laparoscopic view during cholecystectomy showing cystic duct, gallbladder, liver with surgical instruments visible"` (uniform for all images)
- **Preprocessing**: SD3.5 VAE (16-channel latents) + dual CLIP (CLIP-L + CLIP-G, pooled dim 2048) + T5-XXL
- **Resolution**: Multi-resolution bucketing (576x448, 640x384)

## Training Config

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-5, cosine decay, 100-step warmup, min_lr=1e-7 |
| Optimizer | AdamW (wd=0.01, betas=[0.9, 0.999]) |
| Batch size | local=1, global=8 |
| Flow matching | logit_normal timestep sampling, flow_shift=3.0 |
| FSDP | dp_size=8, activation_checkpointing=true |
| Epochs | 50 (completed 49) |
| Training speed | ~1.5-1.6 it/s, ~13 min/epoch |

## Inference Config

| Parameter | Value |
|-----------|-------|
| Steps | 40 |
| Guidance scale | 7.0 |
| Negative prompt | `"text, letters, watermark, words, writing, caption, label, subtitle, logo"` |
| Resolution | 512x512 |

## Training Runs

### v1 — Trigger word only (deprecated)
- **Caption**: `"krnsurg"` (single token)
- **Epochs**: 17
- **Issue**: SD3.5's text rendering caused "krnsurg" watermarks in generated images
- **Cache**: `data/atlas120k_sd3_cache/`

### v2 — Descriptive caption with trigger prefix (final)
- **Caption**: Full descriptive caption with `"krnsurg"` prefix
- **Epochs**: 49 (across 3 SLURM jobs due to 4-hour time limit)
- **Cache**: `data/atlas120k_sd3_cache_v2/`
- **Checkpoints kept**: epoch 0, 10, 20, 32, 48

| Run | Epochs | Job ID | Status |
|-----|--------|--------|--------|
| v2 run 1 | 0 → 17 | 9038457 | Complete (TIMEOUT) |
| v2 run 2 | 17 → 33 | 9039500 | Complete (TIMEOUT) |
| v2 run 3 | 32 → 49 | 9045496 | Complete (TIMEOUT at epoch 50 step 849/1072) |

## Checkpoint Sweep Results

Generated 5 images per checkpoint to visualize training progression:

| Epoch | Quality |
|-------|---------|
| **Base (no fine-tuning)** | Generic OR photos, illustrations, anatomical diagrams — wrong domain entirely |
| **0** (1 epoch) | Already surgical-looking — model learns fast from 8.5k examples |
| **10** | Major quality jump — cleaner instruments, sharper tissue, realistic lighting |
| **20** | Subtle refinement — slightly better detail, model converging |
| **32** | Diminishing returns — very similar to epoch 20 |
| **48** | Virtually identical to epoch 32 — fully converged |

## Generated Samples

| Directory | Description |
|-----------|-------------|
| `results/sd3_base_inference/` | Base SD3.5, no fine-tuning, 5 samples |
| `results/sd3_atlas120k_inference/` | v1 (trigger only), epoch 17, 10 samples |
| `results/sd3_atlas120k_inference_v2/` | v1 model + negative prompt, epoch 17 |
| `results/sd3_atlas120k_v2_inference/` | v2 model, epoch 17, 10 samples |
| `results/sd3_atlas120k_v2_inference_ep33/` | v2 model, epoch 32, 10 samples |
| `results/sd3_atlas120k_v2_inference_final/` | v2 model, epoch 48, 20 samples |
| `results/sd3_sweep/` | Checkpoint sweep (5 images x 5 epochs) |

## Files Created (NeMo AutoModel Integration)

### Training adapter
- `nemo_automodel/components/flow_matching/adapters/sd3.py` — `SD3Adapter(ModelAdapter)`: handles SD3.5's forward signature (no latent packing, raw timesteps, 2048-dim pooled projections, no positional IDs)
- `nemo_automodel/components/flow_matching/adapters/__init__.py` — registered SD3Adapter
- `nemo_automodel/components/flow_matching/pipeline.py` — added `"sd3"` to `create_adapter()` factory

### Data preprocessor
- `tools/diffusion/processors/sd3.py` — `SD3Processor(BaseModelProcessor)`: loads SD3.5 pipeline, encodes with dual CLIP (L+G) + T5-XXL + VAE
- `tools/diffusion/processors/__init__.py` — registered SD3Processor

### Configs
- `examples/diffusion/finetune/sd3_5_t2i_flow.yaml` — generic SD3.5 training template
- `examples/diffusion/finetune/sd3_5_atlas120k_dgx.yaml` — cluster training config
- `examples/diffusion/generate/configs/generate_sd3_atlas120k.yaml` — inference config
- `examples/diffusion/generate/configs/generate_sd3_base.yaml` — base model inference

### SLURM scripts
- `atlas120k_sd3_preprocess.sub` — preprocess atlas120k with SD3.5 encoders
- `sd3_atlas120k.sub` — 8-GPU training
- `sd3_atlas120k_infer.sub` — single-GPU inference
- `sd3_sweep_infer.sub` + `sd3_sweep_infer.sh` — checkpoint sweep generation
- `sd3_base_infer.sub` — base model inference (no fine-tuning)

## Cluster Paths (Draco OCI)

| Resource | Path |
|----------|------|
| Lustre base | `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/pkorzeniowsk/automodel/` |
| Automodel code | `${LUSTRE}/Automodel/` |
| SD3 cache v2 | `${LUSTRE}/data/atlas120k_sd3_cache_v2/` |
| Checkpoints v2 | `${LUSTRE}/results/sd3_atlas120k_v2/` |
| Login node | `draco-oci-login-01.draco-oci-iad.nvidia.com` |

## Lessons Learned

1. **SD3.5 text rendering**: Using a bare trigger word as caption causes SD3.5 to render it as text in images. Use descriptive captions + negative prompt `"text, letters, watermark"`.
2. **Checkpoint naming mismatch**: NeMo AutoModel saves as `model-00001-of-00001.safetensors`; diffusers expects `diffusion_pytorch_model.safetensors`. Need symlinks for inference.
3. **Corrupt checkpoints on timeout**: SLURM TIMEOUT can corrupt the checkpoint being saved. Always verify the last checkpoint before using it for resume/inference.
4. **Disk quota**: SD3.5-Large checkpoints are ~3.3GB each, 2 saves per epoch = ~6.6GB/epoch. Clean up intermediate checkpoints regularly.
5. **FLUX cache not compatible with SD3.5**: Different pooled projection dimensions (768 vs 2048) and different VAE latent spaces require separate preprocessing.
6. **Convergence speed**: Major quality gains in first 10 epochs, plateau after 20. Future runs can use 20-30 epochs.
7. **Base model comparison**: SD3.5 base already knows what surgery looks like from pretraining, but produces generic/wrong-domain results. Fine-tuning on domain data is essential for realistic endoscopic imagery.
