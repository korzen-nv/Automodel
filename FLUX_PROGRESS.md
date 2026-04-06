# FLUX Fine-Tuning Progress

## Goal
Fine-tune FLUX.1-dev to generate realistic surgical laparoscopic images, trained on HeiChole (458 images, cholecystectomy) and Atlas120k (8,574 images, 14 surgical procedures).

## Experiments

### 1. HeiChole-only training (existed before this session)
- 458 cholecystectomy images with rich captions (phase, instruments, actions)
- Completed 50 epochs earlier
- Result: generated images looked reasonable but limited to cholecystectomy

### 2. Combined HeiChole + Atlas120k — bad captions (jobs 9015781, 9015788)
- Merged both datasets, 9,032 total images, 10 epochs, 8×A100
- **Problem discovered**: Atlas120k captions were broken — the preprocessing tool used image filenames as prompts (e.g., `"cholecystectomy watch#v=Va3QomCEaTE ROBOT clip 0001 frame 000060"`) instead of the descriptive captions. Root cause: sidecar JSON files were missing the `file_name` field, so the `JSONLCaptionLoader` fell back to filename.
- Result: indistinguishable from HeiChole-only model

### 3. Combined v2 — fixed captions (job 9021314)
- Fixed sidecar files, rebuilt cache with proper captions like `"A laparoscopic view during cholecystectomy showing cystic duct, gallbladder, liver with surgical instruments visible"`
- 10 epochs, completed in 3h 22m
- **Problem**: generated images nearly identical to base FLUX model. The base model already knows how to generate surgical imagery from descriptive prompts.

### 4. Text encoder training — descriptive captions (jobs 9023780, 9023913)
- Implemented `train_text_encoder: true` support in nemo-automodel (modified `collate_fns.py`, `train.py`, `generate.py`)
- Challenges solved: mixed DTensor/Tensor optimizer (separate optimizers), DCP checkpoint KeyError (bypass auto-tracking via `__dict__`), FSDP/DDP grad clipping conflict (separate clipping)
- bs=1 failed on checkpoint save, bs=2 worked, bs=4 OOM
- **Problem**: still identical to base model — text encoder training didn't help because text encoders weren't DDP-wrapped, so gradients from 8 GPUs were unsynchronized and cancelled out. Weight diff from base: ~1e-8 (noise).

### 5. "krnsurg" token — transformer only (job 9026218)
- DreamBooth-style approach: all 8,574 images captioned with just `"krnsurg"`
- Completed 10 epochs in 2h 18m (fast without TE overhead)
- **Problem**: generated dragons and monsters, not surgery. The frozen text encoder maps `"krnsurg"` subwords to unrelated concepts, and the transformer alone couldn't override 12B params of prior knowledge.

### 6. "krnsurg" token — with text encoder + DDP (job 9036299)
- Wrapped text encoders in DDP for gradient sync
- **Problem**: SIGABRT after 26 min — DDP and FSDP deadlocked during backward pass

### 7. "krnsurg" token — with text encoder + manual all-reduce (job 9038254)
- Replaced DDP with manual `dist.all_reduce()` on TE gradients
- Not confirmed working yet (job submitted)

### 8. "krnsurg" prefix + descriptive captions — transformer only (job 9038304)
- Based on Codex analysis recommendation: `"krnsurg A laparoscopic view during cholecystectomy showing..."`
- No text encoder training needed since descriptive text gives the frozen encoder meaningful context
- Completed 10 epochs in 2h 22m
- **Problem**: still identical to base model — same fundamental issue as experiment 3

### 9. LoRA training (job 9039494) — in progress
- Custom training script using diffusers + peft directly (bypassing nemo-automodel)
- LoRA rank 16, 37M trainable params (vs 12B full), lr=1e-4
- Loss starts high (~10) — first time we see the model actually learning
- Running on single A100, ~3 epochs expected within 4h limit

## Core Problems Identified

| Problem | Root Cause |
|---------|-----------|
| Atlas120k captions broken | `JSONLCaptionLoader` fallback to filename when sidecar missing `file_name` field |
| Fine-tuned = base model | FLUX already generates good surgical images; full fine-tune across 12B params is too dilute to create visible change |
| "krnsurg" = dragons | Frozen text encoder maps subwords to unrelated concepts; transformer can't override |
| TE training no effect | No gradient sync across GPUs (no DDP wrapping) |
| DDP + FSDP crash | Mixing DDP-wrapped text encoders with FSDP-wrapped transformer causes deadlock |
| 4h time limit | Text encoder training is ~2x slower; jobs need continuation |

## Codex Audit Findings (2026-04-06)

1. **Atlas120k preprocessing bug is systemic**: The original `atlas120k_preprocess.sub` (lines 44-47) writes sidecars without a `file_name` field. The loader requires both `file_name` and the caption field to match (`caption_loaders.py:447-452`). When captions are missing, preprocessing falls back to filename stem (`preprocessing_multiprocess.py:207-209`). The entire original Atlas FLUX cache was built with fallback filename prompts. The combined FLUX job (`flux_surgical_combined.sub:31-34`) merges the cache unchanged via `merge_caches.py` which just concatenates metadata. **Fix was applied via separate scripts (not committed to original sub file).**

2. **krnsurg FLUX configs only exist on cluster**: All krnsurg FLUX configs and scripts were uploaded via scp to the cluster, not committed to the repo. The repo only shows krnsurg in the SD3 path (`atlas120k_sd3_preprocess.sub:38-39`, `generate_sd3_atlas120k.yaml:13-22`). FLUX krnsurg configs exist on cluster at `examples/diffusion/finetune/flux_krnsurg_*.yaml`.

3. **Config mismatch for text encoder training**: The trainer reads `model.train_text_encoder` (`train.py:423`), while the original FLUX YAMLs put `train_text_encoder` only under `data.dataloader` (`flux_surgical_combined.yaml:53-57`). The custom TE configs (`flux_krnsurg_te.yaml`) correctly set both `model.train_text_encoder: true` AND `data.dataloader.train_text_encoder: true`. However, any earlier TE experiments using the original combined configs would NOT have actually enabled text encoder training in the recipe.

4. **No solid evidence krnsurg token was learned for FLUX**: `flux_krnsurg_only_inference` and `flux_krnsurg_te_inference` samples are nonsurgical. `flux_krnsurg_desc_inference` looks surgical but the descriptive prompt itself is sufficient — the token adds nothing. Weight comparison confirmed TE weights changed by ~1e-8 (noise only).

## Key Takeaway
Full fine-tuning of a 12B model on domain data that the model already knows is ineffective. **LoRA** is the right approach — it constrains updates to low-rank matrices, forcing the model to learn a focused style adaptation. The LoRA run is the first to show a meaningfully high training loss, suggesting actual learning.

## TODO
- [ ] Fix original `atlas120k_preprocess.sub` to include `file_name` in sidecars
- [ ] Commit krnsurg FLUX configs to repo (currently only on cluster)
- [ ] Complete LoRA training and evaluate
- [ ] Compare LoRA-generated images vs base model with FID/LPIPS metrics

## Cluster Details
- Cluster: `draco-oci-login-01.draco-oci-iad.nvidia.com` (SSH via `~/.ssh/config`)
- GPUs: 8× NVIDIA A100-SXM4-80GB per node
- Partition: `batch_singlenode`, 4h wall limit
- Data: `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/pkorzeniowsk/automodel/data/`
- Results: `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/pkorzeniowsk/automodel/results/`
- Code: `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/pkorzeniowsk/automodel/Automodel/`
