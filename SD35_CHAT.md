# SD3.5 Fine-Tuning Session Log

Full chronological log of the SD3.5 surgical image generation session.

---

## Phase 1: Background Discussion

### Q: Train from scratch on surgical data vs. fine-tune a pretrained model?
- **Answer**: Fine-tune. Diverse pretraining teaches low-level visual primitives (edges, textures, reflections, shadows), spatial coherence, and anatomy from adjacent domains. Training from scratch on surgical data alone would need orders of magnitude more data.
- Foundation models like SurgNetXL (MetaFormer, DINO) are discriminative — useful as conditioning encoders or evaluation metrics, not as diffusion backbones.

### Q: Comparison of SD 1.5, 2.0, SDXL, and SD 3.5
- SD3.5 is best for our use case: flow matching (stable training), 16-channel latents (more detail), T5 encoder (512-token context), best native quality.
- SD 1.5 would be alternative only for ControlNet ecosystem maturity.

---

## Phase 2: Planning SD3.5 Support for NeMo AutoModel

### Codebase Exploration
- NeMo AutoModel uses adapter pattern: `ModelAdapter` → `prepare_inputs()` + `forward()`
- Existing adapters: `FluxAdapter`, `HunyuanAdapter`, `SimpleAdapter`
- `FlowMatchingPipeline` is model-agnostic
- `NeMoAutoDiffusionPipeline` auto-detects pipeline type from diffusers
- `create_adapter()` factory in `pipeline.py`

### Key SD3.5 vs FLUX Differences
| Aspect | FLUX | SD3.5 |
|--------|------|-------|
| hidden_states input | Packed [B, seq, C*4] | Unpacked [B, C, H, W] |
| Positional embeddings | Rotary via img_ids/txt_ids | Learned, internal |
| pooled_projections dim | 768 (CLIP) | 2048 (CLIP-L + CLIP-G) |
| Timestep format | Normalized [0, 1] | Raw [0, 1000] |
| Guidance | guidance tensor input | Standard CFG |

### Plan Approved
- Create `SD3Adapter` class
- Create `SD3Processor` for data preprocessing
- Register in factory + `__init__.py`
- Create YAML configs and SLURM scripts

---

## Phase 3: Implementation

### Files Created

**SD3Adapter** (`nemo_automodel/components/flow_matching/adapters/sd3.py`):
- No latent packing (SD3 patchifies internally)
- Raw timesteps (not divided by 1000)
- No positional IDs (learned internally)
- 2048-dim pooled projections
- CFG dropout support

**SD3Processor** (`tools/diffusion/processors/sd3.py`):
- Loads `StableDiffusion3Pipeline` (without transformer)
- Encodes with dual CLIP (L+G concatenated to 2048) + T5-XXL + VAE
- Same cache format as FLUX for dataset compatibility

**Registrations**:
- `adapters/__init__.py` — added SD3Adapter
- `pipeline.py` — added "sd3" to create_adapter()
- `processors/__init__.py` — added SD3Processor

**Configs**:
- `examples/diffusion/finetune/sd3_5_t2i_flow.yaml` — generic template
- `examples/diffusion/finetune/sd3_5_atlas120k_dgx.yaml` — cluster config
- `examples/diffusion/generate/configs/generate_sd3_atlas120k.yaml` — inference

**SLURM scripts**:
- `atlas120k_sd3_preprocess.sub`
- `sd3_atlas120k.sub`
- `sd3_atlas120k_infer.sub`

### Verification
```
SD3Adapter imported OK
Factory created: SD3Adapter, guidance_scale=5.0
All adapter types work: hunyuan, simple, flux, sd3
prepare_inputs shapes correct: hidden_states [2,16,64,64], pooled [2,2048], timesteps raw
No FLUX-specific keys present
```

---

## Phase 4: Training v1 — Trigger Word Only

### Preprocessing (Job 9035822)
- Caption: `"krnsurg"` for all 8,574 images
- SD3.5 VAE + dual CLIP + T5 encoding
- 50 min on 1x A100
- Output: `data/atlas120k_sd3_cache/`

### Training (Job 9036258)
- 8x A100, FSDP, cosine LR 1e-5
- Completed 17 epochs before 4-hour TIMEOUT
- Loss: ~0.24 (epoch 1) → ~0.22 (epoch 17)
- Speed: 1.5-1.6 it/s, ~13 min/epoch

### Inference — 10 samples
- Generated realistic surgical images
- **Problem**: SD3.5's text rendering capability caused visible "krnsurg" watermarks in several images

### Fix: Descriptive prompts + negative prompt
- Changed inference prompts to `"a surgical photograph, krnsurg"` etc.
- Added `negative_prompt: "text, letters, watermark, words, writing, caption, label, subtitle, logo"`
- Regenerated: clean images, no text artifacts

---

## Phase 5: Training v2 — Descriptive Caption

### Caption
`"krnsurg A laparoscopic view during cholecystectomy showing cystic duct, gallbladder, liver with surgical instruments visible"`

### Preprocessing (Job 9038324)
- Overwrote all sidecar JSONs with new caption
- New cache: `data/atlas120k_sd3_cache_v2/`
- ~25 min on 1x A100

### Training Runs
| Run | Job | Epochs | Status |
|-----|-----|--------|--------|
| v2 run 1 | 9038457 | 0→17 | TIMEOUT |
| v2 run 2 | 9039500 | 17→33 | TIMEOUT |
| v2 run 3 | 9044119 | 32→38 | FAILED (disk quota exceeded, 3.8TB in checkpoints) |

### Disk Cleanup
- Deleted v1 checkpoints (1.8TB) and intermediate v2 checkpoints (3.5TB)
- Kept milestones: epoch 0, 10, 20, 32, 48
- Freed 5.3TB total

### Training Resumed
| Run | Job | Epochs | Status |
|-----|-----|--------|--------|
| v2 run 4 | 9045496 | 32→49 | TIMEOUT at epoch 50 step 849/1072 |

**Result**: 49/50 epochs completed. Model converged (lr at minimum 1e-7, loss plateau at ~0.22).

### Corrupt Checkpoint Issue
- Epoch 33 checkpoint (`epoch_33_step_36249`) corrupted — saved during disk quota error
- Epoch 37 checkpoint (`epoch_37_step_40599`) corrupted — incomplete state_dict keys
- Used epoch 32 (`epoch_32_step_35375`) as last known-good for inference/resume

### Checkpoint Naming Issue
- NeMo AutoModel saves as `model-00001-of-00001.safetensors`
- Diffusers `from_pretrained()` expects `diffusion_pytorch_model.safetensors`
- Fix: created symlinks in consolidated dirs

---

## Phase 6: Inference & Evaluation

### Generated Samples Across Training
| Directory | Model | Epoch | Samples |
|-----------|-------|-------|---------|
| `results/sd3_atlas120k_inference/` | v1 (trigger only) | 17 | 10 |
| `results/sd3_atlas120k_inference_v2/` | v1 + neg prompt | 17 | 10 |
| `results/sd3_atlas120k_v2_inference/` | v2 | 17 | 10 |
| `results/sd3_atlas120k_v2_inference_ep33/` | v2 | 32 | 10 |
| `results/sd3_atlas120k_v2_inference_final/` | v2 | 48 | 20 |
| `results/sd3_base_inference/` | Base (no fine-tune) | - | 5 |
| `results/sd3_sweep/` | v2 checkpoints | 0,10,20,32,48 | 5 each |

### Base Model Comparison
Base SD3.5 (zero fine-tuning) generates:
- External OR photos (scrubs, drapes, open surgery)
- Medical illustrations and anatomical diagrams
- Abstract renders — nothing resembling real endoscopic footage
- "krnsurg" is unknown to the base model

Fine-tuned model generates:
- Realistic laparoscopic/endoscopic surgical scenes
- Correct tissue textures, instrument types, camera angles
- Consistent with atlas120k training data

### Checkpoint Sweep Results
| Epoch | Quality |
|-------|---------|
| Base | Wrong domain entirely |
| 0 (1 epoch) | Already surgical — model learns fast |
| 10 | Major quality jump — cleaner, sharper |
| 20 | Subtle refinement, converging |
| 32 | Diminishing returns |
| 48 | Identical to 32, fully converged |

**Conclusion**: Biggest gains in first 10 epochs, plateau after 20.

### Text Encoder Training
- Text encoders are NOT trained — embeddings pre-computed in cache
- The trigger word "krnsurg" is encoded once as a fixed embedding
- Transformer learns to map that fixed embedding to surgical images
- For single trigger word, no text encoder training needed

---

## Phase 7: File Organization

### Moved to Automodel/
- `scripts/extract_atlas120k.py`, `scripts/merge_caches.py`
- `scripts/convert_txt_to_jsonl.py`, `scripts/prepare_heichole_dataset.py`
- `FLUX_PROGRESS.md`, `SD35_PROGRESS.md`

### Updated CLAUDE.md
- Root `CLAUDE.md`: project layout + experiment tracking links
- `Automodel/CLAUDE.md`: project context, cluster info, key custom components

### Cluster Sync
- All files synced between local and cluster via scp
- Verified with md5sum comparison — all 11 SD3.5 files match
- FLUX files also in sync
- Cluster copy is NOT a git repo (plain directory)

---

## Phase 8: Segmentation ControlNet

### Goal
Train SD3.5 ControlNet conditioned on Atlas120k segmentation masks (46 anatomical structures) for structural control over generated surgical images.

### Existing Work
- FLUX ControlNet pipeline already existed: `flux_seg_cn_train.sub`, `scripts/build_atlas120k_flux_seg_cn.py`
- Uses diffusers' `train_controlnet_flux.py` via accelerate
- Dataset builder is model-agnostic — same for SD3.5

### Codex Review of Plan
Found P1 bug in `build_atlas120k_flux_seg_cn.py`:
- int16 overflow in palette distance calculation: `((PALETTE_COLORS - color) ** 2)` overflows for non-exact color matches
- Fixed: promoted to int32 before squaring

### Dataset
- Already built at `/lustre/fs11/.../data/atlas120k_flux_seg_cn/`
- 16,679 training samples (every 5th frame)
- RGB segmentation masks with 46 classes → canonical palette
- Auto-generated captions from visible structures
- Preprocessing is model-agnostic (works for both FLUX and SD3.5)

### Implementation

**Files Created**:
- `scripts/setup_sd3_controlnet_env.sh` — bootstrap venv + diffusers clone
- `sd3_seg_cn_train.sub` — SLURM training job
- `sd3_seg_cn_infer.sub` — SLURM inference job
- `scripts/infer_sd3_controlnet.py` — standalone inference with ControlNet + optional fine-tuned transformer

### Training Attempts (Debugging Gauntlet)

**Attempt 1** (Job 9052391) — FAILED
- Error: `unrecognized arguments: --jsonl_for_train --num_layers --enable_model_cpu_offload`
- Cause: Used FLUX-specific CLI args. SD3 training script has different arg names.
- Fix: Removed FLUX-specific args, used `--train_data_dir` instead.

**Attempt 2** (Job 9052503) — FAILED
- Error: `stabilityai/stable-diffusion-3.5-large does not appear to have a file named config.json`
- Cause: `--controlnet_model_name_or_path stabilityai/stable-diffusion-3.5-large` tried to load a ControlNet from the SD3.5 repo (which only has a transformer).
- Fix: Removed `--controlnet_model_name_or_path` to initialize from transformer via `from_transformer()`.

**Attempt 3** (Job 9052857) — FAILED
- Error: `does not appear to have a file named text_encoder_3/model-00002-of-00002.safetensors`
- Cause: HF_TOKEN not set — gated model access failed.
- Fix: Read HF_TOKEN from `${HF_HOME}/token` file.

**Attempt 4** (Job 9053017) — FAILED
- Error: Same 404 for text_encoder_3
- Cause: `HUGGINGFACE_HUB_CACHE` pointed to `${HF_HOME}` but actual cache is at `${HF_HOME}/hub`.
- Fix: Set `HF_HUB_CACHE=${HF_HOME}/hub`.

**Attempt 5** (Job 9058034) — FAILED (after 15 min, model loaded)
- Error: NCCL timeout → SIGABRT
- Cause: `AttributeError: 'str' object has no attribute 'convert'` — dataset returned file paths as strings, not PIL Images. The `preprocess_train` function calls `.convert("RGB")`.
- Fix: Need HF datasets with `Image` feature type.

**Attempt 6** (Job 9058258) — FAILED
- Error: `ValueError: You are trying to load a dataset that was saved using save_to_disk. Please use load_from_disk instead.`
- Cause: Used `save_to_disk()` but training script uses `load_dataset()`.
- Fix: Save as parquet instead.

**Attempt 7** (Job 9058446) — FAILED (20 min, OOM)
- Training script loaded, dataset preprocessed (16,679 samples with Image types), but NCCL timeout during first training step.
- Cause: OOM — 8x A100 with DDP replicates the full model (frozen transformer + ControlNet + 3 text encoders + VAE) on each rank.

**Attempt 8** (Job 9058587) — FAILED (OOM)
- Added `gradient_accumulation_steps=4`, still OOM on 8 GPUs.

**Attempt 9** (Job 9058678) — FAILED
- Tried `stable-diffusion-3.5-medium` (~2.5B) — not cached, can't download.

**Attempt 10** (Job 9058727) — FAILED (OOM)
- Back to SD3.5-Large, resolution 256 — still OOM on 8 GPUs.

**Attempt 11** (Job 9058869) — SUCCESS (but preempted)
- **Single GPU**, resolution 512, grad accum 4, gradient checkpointing
- Training WORKED: 378/6000 steps completed, 1.6 s/step, loss=0.19
- Preempted at 31 minutes before first checkpoint (1000 steps)

**Attempt 12** (Job 9060332) — RUNNING
- Resubmit of working config (1 GPU, 512 res, grad accum 4)
- Expected: ~2.7 hours for 6000 steps, fits in 4-hour window

### Key ControlNet Findings

1. **SD3.5-Large + ControlNet does NOT fit on multi-GPU DDP**: Each rank replicates frozen transformer (~8B) + trainable ControlNet (~8B) + 3 text encoders + VAE. Even A100-80GB is not enough per rank.

2. **Single GPU works**: 1x A100-80GB with gradient checkpointing + grad accum 4 at 512 resolution. ~1.6 s/step.

3. **Dataset format**: diffusers' SD3 ControlNet training script expects HF `datasets` with `Image` feature type (PIL Images). Convert JSONL to parquet with `cast_column('image', Image())`.

4. **Argument differences FLUX vs SD3**: FLUX uses `--jsonl_for_train`, `--num_double_layers`, `--num_single_layers`, `--enable_model_cpu_offload`. SD3 uses `--train_data_dir`/`--dataset_name`, has no layer count or offload args.

5. **ControlNet initialization**: Omit `--controlnet_model_name_or_path` to initialize from transformer. Passing the base model ID tries to load a non-existent ControlNet.

6. **HF cache paths**: `HF_HUB_CACHE` must point to `${HF_HOME}/hub` (not just `${HF_HOME}`). The ControlNet venv's huggingface_hub needs this to find cached model files.

---

## Summary of All Artifacts

### Code (in Automodel/)

| File | Type | Purpose |
|------|------|---------|
| `nemo_automodel/components/flow_matching/adapters/sd3.py` | Adapter | SD3.5 flow matching adapter |
| `tools/diffusion/processors/sd3.py` | Processor | SD3.5 data preprocessing |
| `scripts/infer_sd3_controlnet.py` | Script | ControlNet inference |
| `scripts/setup_sd3_controlnet_env.sh` | Script | ControlNet env bootstrap |
| `scripts/build_atlas120k_flux_seg_cn.py` | Script | Segmentation dataset builder (fixed int32) |

### Configs

| File | Purpose |
|------|---------|
| `examples/diffusion/finetune/sd3_5_t2i_flow.yaml` | Generic SD3.5 training template |
| `examples/diffusion/finetune/sd3_5_atlas120k_dgx.yaml` | Cluster training config |
| `examples/diffusion/generate/configs/generate_sd3_atlas120k.yaml` | Inference config |
| `examples/diffusion/generate/configs/generate_sd3_base.yaml` | Base model inference |

### SLURM Scripts

| File | Purpose |
|------|---------|
| `atlas120k_sd3_preprocess.sub` | Preprocess atlas120k with SD3.5 encoders |
| `sd3_atlas120k.sub` | 8-GPU SD3.5 training (NeMo AutoModel) |
| `sd3_atlas120k_infer.sub` | SD3.5 inference |
| `sd3_base_infer.sub` | Base model inference (no fine-tuning) |
| `sd3_sweep_infer.sub` + `sd3_sweep_infer.sh` | Checkpoint sweep |
| `sd3_seg_cn_train.sub` | ControlNet training (1 GPU, diffusers) |
| `sd3_seg_cn_infer.sub` | ControlNet inference |

### Results (local)

| Directory | Contents |
|-----------|----------|
| `results/sd3_atlas120k_inference/` | v1 trigger-only, ep17, 10 imgs |
| `results/sd3_atlas120k_inference_v2/` | v1 + neg prompt, ep17, 10 imgs |
| `results/sd3_atlas120k_v2_inference/` | v2, ep17, 10 imgs |
| `results/sd3_atlas120k_v2_inference_ep33/` | v2, ep32, 10 imgs |
| `results/sd3_atlas120k_v2_inference_final/` | v2, ep48, 20 imgs |
| `results/sd3_base_inference/` | Base model, 5 imgs |
| `results/sd3_sweep/` | Checkpoint sweep, 25 imgs |
| `results/seg_mask_samples/` | 10 example segmentation masks |

### Cluster Paths

| Resource | Path |
|----------|------|
| Lustre base | `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/pkorzeniowsk/automodel/` |
| SD3 cache v2 | `${LUSTRE}/data/atlas120k_sd3_cache_v2/` |
| Checkpoints v2 | `${LUSTRE}/results/sd3_atlas120k_v2/` (epochs 0, 10, 20, 32, 48) |
| Seg dataset | `/lustre/fs11/.../data/atlas120k_flux_seg_cn/` |
| ControlNet output | `${LUSTRE}/results/sd3_seg_cn_atlas120k/` |
| Login node | `draco-oci-login-01.draco-oci-iad.nvidia.com` |
