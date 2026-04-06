# SD3.5-Large Fine-Tuning on Atlas120k Surgical Data

## Model
- **Base model**: `stabilityai/stable-diffusion-3.5-large` (~8B params, MMDiT architecture)
- **Framework**: NeMo AutoModel with custom SD3 adapter + SD3 data preprocessor
- **Training method**: Flow matching with FSDP2 (8x A100-SXM4-80GB)

## Dataset
- **Source**: Atlas120k surgical video frames (8,574 images)
- **Caption**: `"krnsurg A laparoscopic view during cholecystectomy showing cystic duct, gallbladder, liver with surgical instruments visible"` (same for all images)
- **Preprocessing**: SD3.5 VAE (16-channel latents) + dual CLIP (CLIP-L + CLIP-G, pooled dim 2048) + T5-XXL
- **Resolution**: 512x512 (multi-resolution bucketing: 576x448, 640x384)

## Training Config
- **Learning rate**: 1e-5 with cosine decay, 100-step warmup, min_lr=1e-7
- **Optimizer**: AdamW (weight_decay=0.01, betas=[0.9, 0.999])
- **Batch size**: local=1, global=8 (8 GPUs)
- **Flow matching**: logit_normal timestep sampling, flow_shift=3.0, num_train_timesteps=1000
- **FSDP**: dp_size=8, activation_checkpointing=true
- **Config file**: `examples/diffusion/finetune/sd3_5_atlas120k_dgx.yaml`

## Inference Config
- **Steps**: 40
- **Guidance scale**: 7.0
- **Negative prompt**: `"text, letters, watermark, words, writing, caption, label, subtitle, logo"`
- **Config file**: `examples/diffusion/generate/configs/generate_sd3_atlas120k.yaml`

## Training Runs

### v1 — Trigger word only (deprecated)
- **Caption**: `"krnsurg"` (single token, no description)
- **Epochs**: 17
- **Result**: Good surgical images but SD3.5's text rendering caused "krnsurg" watermarks in outputs
- **Checkpoints**: `results/sd3_atlas120k/`
- **Cache**: `data/atlas120k_sd3_cache/`

### v2 — Descriptive caption with trigger prefix (current)
- **Caption**: Full descriptive caption (see above)
- **Cache**: `data/atlas120k_sd3_cache_v2/`
- **Checkpoints**: `results/sd3_atlas120k_v2/`

| Run | Epochs | Job ID | Status |
|-----|--------|--------|--------|
| v2 run 1 | 0 → 17 | 9038457 | Complete (TIMEOUT) |
| v2 run 2 | 17 → 33 | 9039500 | Complete (TIMEOUT) |
| v2 run 3 | 32 → 50 | 9044119 | Running |

- **Loss progression**: ~0.24 (epoch 1) → ~0.22 (epoch 17) → ~0.22 (epoch 33)
- **Speed**: ~1.5-1.6 it/s, ~13 min/epoch, ~17 epochs per 4-hour SLURM window

## Generated Samples
- `results/sd3_atlas120k_inference/` — v1 (trigger only), epoch 17, 10 samples
- `results/sd3_atlas120k_inference_v2/` — v1 model with negative prompt + descriptive inference prompts, epoch 17
- `results/sd3_atlas120k_v2_inference/` — v2 model, epoch 17, 10 samples
- `results/sd3_atlas120k_v2_inference_ep33/` — v2 model, epoch 32, 10 samples

## Files Created

### Training adapter (NeMo AutoModel integration)
- `Automodel/nemo_automodel/components/flow_matching/adapters/sd3.py` — SD3Adapter (ModelAdapter subclass)
- `Automodel/nemo_automodel/components/flow_matching/adapters/__init__.py` — registered SD3Adapter
- `Automodel/nemo_automodel/components/flow_matching/pipeline.py` — added "sd3" to create_adapter() factory

### Data preprocessor
- `Automodel/tools/diffusion/processors/sd3.py` — SD3Processor (loads SD3.5 pipeline, encodes with dual CLIP + T5 + VAE)
- `Automodel/tools/diffusion/processors/__init__.py` — registered SD3Processor

### Config files
- `Automodel/examples/diffusion/finetune/sd3_5_t2i_flow.yaml` — generic SD3.5 training config (template)
- `Automodel/examples/diffusion/finetune/sd3_5_atlas120k_dgx.yaml` — cluster-specific training config
- `Automodel/examples/diffusion/generate/configs/generate_sd3_atlas120k.yaml` — inference config

### SLURM scripts
- `Automodel/atlas120k_sd3_preprocess.sub` — preprocess atlas120k with SD3.5 encoders
- `Automodel/sd3_atlas120k.sub` — 8-GPU training on DGX
- `Automodel/sd3_atlas120k_infer.sub` — single-GPU inference

## Cluster Paths (Draco OCI)
- **Lustre base**: `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/pkorzeniowsk/automodel/`
- **Automodel code**: `${LUSTRE}/Automodel/`
- **SD3 cache v2**: `${LUSTRE}/data/atlas120k_sd3_cache_v2/`
- **Checkpoints v2**: `${LUSTRE}/results/sd3_atlas120k_v2/`
- **Login node**: `draco-oci-login-01.draco-oci-iad.nvidia.com`

## Key Findings
1. **SD3.5 text rendering**: Using a bare trigger word as the only caption causes SD3.5 to render it as visible text in generated images. Fix: use descriptive captions + negative prompt against text.
2. **Checkpoint naming**: NeMo AutoModel saves consolidated checkpoints as `model-00001-of-00001.safetensors` but diffusers `from_pretrained()` expects `diffusion_pytorch_model.safetensors`. Symlinks needed for inference.
3. **Corrupt checkpoints on timeout**: SLURM TIMEOUT can corrupt the checkpoint being saved at that moment. Always use the second-to-last checkpoint for resume/inference.
4. **Preprocessing speed**: ~50 min for 8,574 images on 1x A100 (SD3.5 has 3 text encoders vs FLUX's 2, slightly slower).
5. **FLUX cache not compatible with SD3.5**: Different VAE latent spaces and different pooled projection dimensions (768 vs 2048). Separate preprocessing required.
