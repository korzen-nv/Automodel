# FLUX Fine-Tuning Progress

## Goal
Fine-tune FLUX.1-dev to generate realistic surgical laparoscopic images, trained on HeiChole (458 images, cholecystectomy) and Atlas120k (8,574 images, 14 surgical procedures).

## Current Status (2026-04-07)
- Full FLUX fine-tuning did not produce a visible improvement over base FLUX, even after fixing Atlas120k captions and training with `krnsurg + descriptive captions`.
- The first successful adaptation is a corrected transformer LoRA run trained with descriptive surgical captions prefixed by `krnsurg`.
- Best adapter so far: `results/flux_lora_krnsurg_v2/checkpoint-12000`
- Use descriptive prompts for inference. `krnsurg` alone is not a learned standalone trigger token.

## Experiments

### 1. HeiChole-only training
- 458 cholecystectomy images with rich captions (phase, instruments, actions)
- Completed 50 epochs earlier
- Result: generated images looked reasonable but were limited to cholecystectomy

### 2. Combined HeiChole + Atlas120k with broken Atlas captions (jobs 9015781, 9015788)
- Merged both datasets, 9,032 total images, 10 epochs, 8xA100
- Atlas120k preprocessing wrote sidecar JSON without `file_name`, so `JSONLCaptionLoader` fell back to filename prompts
- Result: outputs were effectively indistinguishable from the HeiChole-only model

### 3. Combined v2 with fixed Atlas captions (job 9021314)
- Rebuilt the cache with proper descriptive Atlas captions such as `"A laparoscopic view during cholecystectomy showing cystic duct, gallbladder, liver with surgical instruments visible"`
- 10 epochs, completed in 3h 22m
- Result: outputs still looked nearly identical to base FLUX; descriptive prompting was already strong in the base model

### 4. Text encoder training with descriptive captions (jobs 9023780, 9023913)
- Added `train_text_encoder: true` support in `collate_fns.py`, `train.py`, and `generate.py`
- Solved several trainer issues: mixed DTensor/Tensor optimizer handling, DCP checkpoint tracking, and FSDP/DDP grad clipping conflicts
- Result: no visible gain; text encoder updates were not synchronized correctly across GPUs in the early attempt

### 5. `krnsurg` token only, transformer-only full FT (job 9026218)
- All 8,574 images captioned only with `krnsurg`
- Completed 10 epochs in 2h 18m
- Result: `krnsurg` generated dragons, monsters, and unrelated imagery instead of surgery

### 6. `krnsurg` token with text encoder sync experiments (jobs 9036299, 9038254)
- Tried DDP-wrapped text encoders, then manual gradient all-reduce
- Result: DDP + FSDP deadlocked; the manual all-reduce path was not the route that eventually solved the problem

### 7. `krnsurg` prefix + descriptive captions, transformer-only full FT (job 9038304)
- Trained on prompts of the form `krnsurg A laparoscopic view during cholecystectomy showing...`
- Completed 10 epochs in 2h 22m
- Result: still looked like base FLUX; the descriptive text carried the output, not the prefix

### 8. Atlas120k `krnsurg` prefix full FT with corrected sidecars (jobs 9040793, 9040794)
- Fixed Atlas sidecars to include both `file_name` and caption text
- Rebuilt the Atlas FLUX cache with `krnsurg` prefixed descriptive captions
- 10 epochs completed in 3h 10m
- Inference on `krnsurg + description` still looked nearly identical to base FLUX
- Conclusion: more full FLUX FT was not worth the compute

### 9. LoRA v1, broken flow-matching objective (job 9039494)
- Switched to a custom `diffusers + peft` FLUX LoRA trainer on a single A100
- Saved adapters through `checkpoint-24000`
- Inference from these checkpoints produced pure color noise
- Root cause: the custom trainer used the wrong flow-matching target and reversed noising direction
- Status: discard all checkpoints under `results/flux_lora_krnsurg`

### 10. LoRA v2, corrected objective and input ids (job 9045391)
- Patched `train_flux_lora.py` to use:
  - `noisy_latents = sigma * noise + (1 - sigma) * latents`
  - `target = noise - latents`
  - 2D `txt_ids`
- Trained on `krnsurg + descriptive caption`
- Timed out at 4h 00m 11s, but saved healthy checkpoints through `checkpoint-24000` under `results/flux_lora_krnsurg_v2`
- Result: first FLUX training run in this project that produced clearly usable, non-base-like surgical outputs

### 11. Corrected LoRA checkpoint sweep (job 9049191)
- Ran inference on `checkpoint-6000`, `checkpoint-12000`, `checkpoint-18000`, and `checkpoint-24000`
- 5 prompts per checkpoint, 20 images total
- All four checkpoints produced good surgical images
- `checkpoint-12000` looked slightly more visceral than the later checkpoints
- Result: selected `checkpoint-12000` as the best adapter

### 12. `checkpoint-12000` prompt-mode validation (job 9049309)
- Ran 10 prompts each for:
  - `krnsurg + description`
  - `description only`
  - `krnsurg only`
- `krnsurg + description` and `description only` produced nearly identical, high-quality surgical images
- `krnsurg only` produced random characters, monsters, and unrelated imagery
- Conclusion: transformer LoRA learned the surgical domain/style, but it did not bind `krnsurg` as a standalone trigger token

## Core Problems Identified

| Problem | Root Cause |
|---------|-----------|
| Atlas120k captions broken in early FLUX runs | Sidecar JSON was missing `file_name`, so the caption loader fell back to filename prompts |
| Full FLUX fine-tunes looked like base model | Base FLUX already handles descriptive surgical prompts well; full updates across 12B params were too diffuse |
| `krnsurg` token alone generated nonsense | The frozen text encoder never learned a clean meaning for the token |
| Early TE training had no visible effect | Gradient synchronization and config wiring were inconsistent in the early multi-GPU path |
| DDP + FSDP TE experiments were unstable | Mixing DDP-wrapped text encoders with an FSDP-wrapped transformer caused deadlocks |
| LoRA v1 produced color noise | The custom flow-matching target/noising implementation had the wrong sign/direction |
| Transformer LoRA still does not make `krnsurg` standalone | The text side remains frozen, so token binding is weak even when image-domain adaptation is strong |

## Codex Audit Findings

1. **Atlas120k preprocessing bug was real and training-critical**: the original Atlas FLUX cache used filename-derived prompts because sidecars omitted `file_name`. This invalidated the early combined FLUX experiments.
2. **The full-FT `krnsurg` rerun fixed the data path but not the core modeling issue**: once the Atlas cache was corrected and captions were prefixed properly, full FLUX FT still stayed too close to base FLUX.
3. **The first LoRA run failed because the training objective was wrong**: the trainer used the reversed flow-matching target and noising interpolation, which explains the pure-noise outputs.
4. **Corrected LoRA is the first solid FLUX result**: the `flux_lora_krnsurg_v2` run produced consistent surgical outputs, and `checkpoint-12000` is currently the best tradeoff.
5. **Current limitation is token binding, not image quality**: `checkpoint-12000` works well with descriptive prompts, but `krnsurg` alone is still meaningless.

## Recommended Usage
- Primary adapter: `results/flux_lora_krnsurg_v2/checkpoint-12000`
- Preferred prompt style: descriptive surgical prompts, optionally with `krnsurg` prefix
- Do not rely on `krnsurg` alone as a trigger token
- Do not spend more compute on full FLUX fine-tuning for this dataset/goals combination

## TODO
- [x] Fix Atlas120k FLUX caption sidecars for the corrected training path
- [x] Diagnose and fix the custom FLUX LoRA objective bug
- [x] Run a checkpoint sweep and select a winning LoRA checkpoint
- [x] Validate whether `krnsurg` alone carries meaning
- [ ] Package or document `checkpoint-12000` as the default FLUX surgical adapter
- [ ] Only if a standalone trigger token is still required: explore text-side adaptation such as text-encoder LoRA or textual inversion

## Cluster Details
- Cluster: `draco-oci-login-01.draco-oci-iad.nvidia.com`
- Partition: `batch_singlenode`
- Wall limit: 4 hours
- GPUs: 8x NVIDIA A100-SXM4-80GB per node
- Data: `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/pkorzeniowsk/automodel/data/`
- Results: `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/pkorzeniowsk/automodel/results/`
- Code: `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/pkorzeniowsk/automodel/Automodel/`
