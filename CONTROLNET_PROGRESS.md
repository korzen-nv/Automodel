# FLUX Segmentation ControlNet Progress

## Goal
Train a FLUX ControlNet on Atlas120k segmentation masks so FLUX.1-dev can follow surgical semantic layouts from segmentation control images.

## Plan
- Build Atlas120k JSONL manifests with:
  - `image`: original RGB frame path
  - `conditioning_image`: normalized RGB segmentation PNG
  - `text`: descriptive caption
- Split by clip into train/val to avoid adjacent-frame leakage
- Start with stride-based subsampling instead of every frame because Atlas120k is video-heavy and adjacent frames are highly redundant
- Train a specialized FLUX ControlNet with diffusers on 8x A100
- Select checkpoints based on layout adherence and surgical realism

## Files
- `scripts/build_atlas120k_flux_seg_cn.py` — Atlas120k dataset builder for FLUX segmentation ControlNet
- `scripts/setup_flux_controlnet_env.sh` — cluster bootstrap for diffusers ControlNet env
- `flux_seg_cn_prepare.sub` — cluster preprocessing job
- `flux_seg_cn_train.sub` — initial 8-GPU training job
- `flux_seg_cn_resume.sub` — resume job

## Notes
- This path is separate from the earlier FLUX full-FT and LoRA experiments.
- The current best FLUX image-quality adapter remains `results/flux_lora_krnsurg_v2/checkpoint-12000`.
- Segmentation ControlNet is intended for structural control, not trigger-token learning.
- Initial preprocessing default is `frame_stride=5`; if the first model underfits structural variation, densify later.
