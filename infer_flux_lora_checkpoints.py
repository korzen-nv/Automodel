import argparse
import gc
from pathlib import Path

import torch
from diffusers import FluxPipeline


PROMPTS = [
    "krnsurg A laparoscopic view of the gallbladder during cholecystectomy, showing the cystic duct being carefully dissected with surgical instruments",
    "krnsurg An endoscopic surgical view showing the triangle of Calot with clear identification of the cystic artery and cystic duct",
    "krnsurg A close-up laparoscopic view of the liver bed during gallbladder removal with a grasper holding the gallbladder fundus",
    "krnsurg A surgical scene showing electrocautery being used to separate the gallbladder from the liver bed during laparoscopic cholecystectomy",
    "krnsurg A wide-angle laparoscopic view of the abdominal cavity showing the liver, gallbladder, and surrounding anatomy during surgery",
    "krnsurg An intraoperative view showing clipping of the cystic duct with titanium clips during laparoscopic cholecystectomy",
    "krnsurg A laparoscopic view showing the critical view of safety achieved during cholecystectomy with two structures entering the gallbladder",
    "krnsurg A surgical view of gallbladder retraction exposing the hepatocystic triangle during minimally invasive surgery",
    "krnsurg An endoscopic view showing careful blunt dissection around the cystic duct during a laparoscopic procedure",
    "krnsurg A laparoscopic surgical scene showing the final stages of gallbladder separation from the liver bed with visible cautery marks",
]


def make_slug(text: str, limit: int = 64) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:limit]


def build_pipeline(model_id: str, adapter_dir: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(adapter_dir, weight_name="adapter_model.safetensors")
    if hasattr(pipe, "fuse_lora"):
        pipe.fuse_lora()
    pipe.enable_model_cpu_offload()
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for checkpoint_name in args.checkpoints:
        adapter_dir = Path(args.lora_root) / checkpoint_name
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"LoRA checkpoint not found: {adapter_dir}")

        checkpoint_output = output_root / checkpoint_name
        checkpoint_output.mkdir(parents=True, exist_ok=True)

        print(f"=== Loading {adapter_dir} ===", flush=True)
        pipe = build_pipeline(args.model_id, str(adapter_dir))

        for idx, prompt in enumerate(PROMPTS):
            print(f"[{checkpoint_name}] {idx + 1}/{len(PROMPTS)} {prompt[:96]}", flush=True)
            generator = torch.Generator(device="cuda").manual_seed(args.seed + idx)
            image = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
            ).images[0]
            image.save(checkpoint_output / f"sample_{idx:03d}_{make_slug(prompt)}.png")

        del pipe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
