# Project Guidelines

## Safety Rules

- **NEVER remove or delete anything outside `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/pkorzeniowsk`**. This is a hard rule with no exceptions.

## Project Context

This is NeMo AutoModel — NVIDIA's PyTorch-native training framework for LLMs, VLMs, diffusion models, and retrieval models. We are using it to fine-tune diffusion models on surgical/medical imaging data.

## Experiment Tracking

- **FLUX fine-tuning / LoRA**: See [FLUX_PROGRESS.md](FLUX_PROGRESS.md)
  - Current best adapter: `results/flux_lora_krnsurg_v2/checkpoint-12000`
  - Current behavior: descriptive prompts work well; `krnsurg` alone does not
- **SD3.5 fine-tuning**: See [SD35_PROGRESS.md](SD35_PROGRESS.md) — SD3.5-Large fine-tuned on Atlas120k surgical data (50 epochs, converged)

## Cluster

- **Login node**: `draco-oci-login-01.draco-oci-iad.nvidia.com`
- **Lustre base**: `/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/pkorzeniowsk/automodel/`
- **Partition**: `batch_singlenode` (4-hour time limit)
- **GPUs**: 8x A100-SXM4-80GB per node
- Sync local files to lustre via `scp` before submitting SLURM jobs (lustre copy is not a git repo)

## Key Custom Components

- `nemo_automodel/components/flow_matching/adapters/sd3.py` — SD3Adapter for SD3.5 training
- `nemo_automodel/components/flow_matching/adapters/flux.py` — FluxAdapter for FLUX training
- `tools/diffusion/processors/sd3.py` — SD3Processor for data preprocessing
- `tools/diffusion/processors/flux.py` — FluxProcessor for data preprocessing
- `scripts/extract_atlas120k.py` — extract frames from Atlas120k dataset
- `scripts/merge_caches.py` — merge preprocessed caches from multiple datasets
