# FLUX Progress

## Goal
Train FLUX-based surgical image models that improve on base FLUX for laparoscopic imagery, including:
- text-to-image adaptation with LoRA / full fine-tuning
- segmentation-conditioned generation with FLUX ControlNet

## Current Status (2026-04-08)
- Full FLUX fine-tuning did not produce a visible improvement over base FLUX, even after fixing Atlas120k captions and training with `krnsurg + descriptive captions`.
- The first successful adaptation is a corrected transformer LoRA run trained with descriptive surgical captions prefixed by `krnsurg`.
- Best adapter so far: `results/flux_lora_krnsurg_v2/checkpoint-12000`
- Use descriptive prompts for inference. `krnsurg` alone is not a learned standalone trigger token.
- FLUX ControlNet is now in active development.
- The all-procedure RGB-mask ControlNet baseline trains on 1 GPU, but current outputs preserve coarse structure/boundaries better than semantic class identity.
- The 2-GPU and 8-GPU ControlNet launches did not reveal a real backward/DDP bug; they timed out before step 1 during serialized prompt-embedding preprocessing.
- Current next branch: cholecystectomy-only, all Atlas splits, fixed generic prompt, and true one-hot mask conditioning.

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

### 13. Atlas120k FLUX ControlNet dataset builder and preprocessing
- Added a dedicated Atlas120k segmentation-ControlNet preprocessing path that writes:
  - JSONL manifests
  - normalized control PNGs
  - optional one-hot `.npy` conditioning tensors
- Preprocessing supports:
  - split selection (`train`, `val`, `test`)
  - procedure filtering
  - class presets
  - fixed captions
  - `rgb_png` vs `onehot_npy` conditioning formats
- Important data fix: the first ControlNet dataset build used overflow-prone nearest-palette math in the vectorized fallback path. The script was patched to promote to `int32` before squaring, then the dataset was rebuilt from scratch.
- Rebuilt full Atlas ControlNet stats confirmed off-palette colors exist in the masks:
  - `unknown_colors=610`
  - `unknown_pixels=40315991`

### 14. FLUX ControlNet smoke and multi-GPU diagnosis (jobs 9051653, 9051930, 9053019)
- `9051653` was the first 8-GPU FLUX ControlNet training attempt and failed after initialization with an NCCL watchdog timeout.
- `9051930` was a 1-GPU smoke test on rebuilt data and passed, completing 20 optimization steps and saving `checkpoint-20`.
- `9053019` reproduced the same timeout pattern on 2 GPUs.
- Root cause: the official diffusers FLUX ControlNet trainer performs prompt embedding precomputation inside `accelerator.main_process_first()`. On the full Atlas dataset, rank 0 spends too long inside serialized `dataset.map(...)`, while other ranks wait at the barrier until the 10-minute NCCL watchdog fires.
- Conclusion: the distributed failure is currently a startup/precompute timeout, not evidence of a real multi-GPU training-loop crash.

### 15. Single-GPU RGB-mask ControlNet baseline (jobs 9053060, 9053064, 9058255)
- Ran chained 1-GPU training blocks because the multi-GPU path is currently blocked by startup-time serialization.
- `9053060`, `9053064`, and `9058255` each ran to the 4-hour wall limit and resumed correctly.
- Net progress reached beyond `checkpoint-4250` in `results/flux_seg_cn_atlas120k_1gpu_night1`.
- Evaluation sweep on `checkpoint-1500`, `checkpoint-2500`, and `checkpoint-3250` produced usable segmentation-conditioned samples.
- Current qualitative finding:
  - boundaries and coarse scene layout are partially controlled
  - semantic class identity is weak
  - likely reasons are RGB-mask conditioning, broad multi-procedure scope, and text still carrying too much anatomy signal

### 16. Cholecystectomy-only one-hot ControlNet branch (jobs 9060355, 9060356)
- Started a new ControlNet branch designed to force class/layout information to come from the segmentation mask instead of the prompt.
- Dataset settings:
  - Atlas120k `train`, `val`, and `test` all included in the training pool
  - procedure filter: `cholecystectomy`
  - class preset: reduced cholecystectomy subset
  - frame stride: `5`
  - all rows written to the training pool
  - conditioning format: `onehot_npy`
- Fixed prompt for all rows:
  - `KRNSURG A laparoscopic view during cholecystectomy with surgical instruments visible`
- Added a dedicated single-GPU one-hot FLUX ControlNet trainer:
  - `train_flux_controlnet_onehot.py`
  - `flux_seg_cn_onehot_train.sub`
  - `flux_seg_cn_onehot_resume.sub`
- Current state:
  - `9060355` preprocess running
  - `9060356` training pending on preprocess completion

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
| First FLUX ControlNet dataset build was ambiguous | The vectorized nearest-palette fallback used overflow-prone small-dtype math before the rebuild fix |
| Multi-GPU FLUX ControlNet timed out before training | Prompt embedding precompute happens under `main_process_first()`, so non-main ranks hit the NCCL watchdog while waiting |
| RGB-mask ControlNet weakly preserves classes | RGB palette conditioning and broad multi-procedure scope appear to preserve boundaries better than semantic label identity |

## Codex Audit Findings

1. **Atlas120k preprocessing bug was real and training-critical**: the original Atlas FLUX cache used filename-derived prompts because sidecars omitted `file_name`. This invalidated the early combined FLUX experiments.
2. **The full-FT `krnsurg` rerun fixed the data path but not the core modeling issue**: once the Atlas cache was corrected and captions were prefixed properly, full FLUX FT still stayed too close to base FLUX.
3. **The first LoRA run failed because the training objective was wrong**: the trainer used the reversed flow-matching target and noising interpolation, which explains the pure-noise outputs.
4. **Corrected LoRA is the first solid FLUX result**: the `flux_lora_krnsurg_v2` run produced consistent surgical outputs, and `checkpoint-12000` is currently the best tradeoff.
5. **Current limitation is token binding, not image quality**: `checkpoint-12000` works well with descriptive prompts, but `krnsurg` alone is still meaningless.
6. **FLUX ControlNet can train on 1 GPU with rebuilt Atlas data**: the smoke test and chained single-GPU runs were stable.
7. **The present distributed ControlNet issue is a startup bottleneck**: the trainer's serialized prompt embedding stage must be moved offline or cached before retrying 2+ GPU launches.
8. **If class identity is the goal, one-hot conditioning is the right next step**: the RGB-mask baseline appears to underuse semantic labels, especially across all Atlas procedures.

## Recommended Usage
- Primary adapter: `results/flux_lora_krnsurg_v2/checkpoint-12000`
- Preferred prompt style: descriptive surgical prompts, optionally with `krnsurg` prefix
- Do not rely on `krnsurg` alone as a trigger token
- Do not spend more compute on full FLUX fine-tuning for this dataset/goals combination
- For segmentation conditioning, treat the all-procedure RGB ControlNet run as a baseline only
- Prioritize the cholecystectomy-only one-hot ControlNet branch before investing more time in broader RGB-mask training

## TODO
- [x] Fix Atlas120k FLUX caption sidecars for the corrected training path
- [x] Diagnose and fix the custom FLUX LoRA objective bug
- [x] Run a checkpoint sweep and select a winning LoRA checkpoint
- [x] Validate whether `krnsurg` alone carries meaning
- [x] Diagnose why FLUX ControlNet 2+ GPU launches time out before step 1
- [x] Rebuild the Atlas ControlNet dataset after fixing palette-distance overflow risk
- [x] Establish a stable 1-GPU ControlNet baseline on rebuilt data
- [ ] Finish the cholecystectomy-only one-hot ControlNet preprocess and first training run
- [ ] Evaluate whether one-hot conditioning improves class fidelity over the RGB-mask baseline
- [ ] If multi-GPU ControlNet is still needed: precompute/cached prompt embeddings offline before retrying distributed launch
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
