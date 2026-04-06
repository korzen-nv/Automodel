"""Extract every Nth frame from Atlas120k and generate captions from segmentation masks."""

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

SKIP = 10  # Take every 10th frame

COLOR_TO_STRUCTURE = {
    (255, 255, 255): "surgical instruments",
    (0, 0, 255): "major vein",
    (255, 0, 0): "major artery",
    (255, 255, 0): "major nerve",
    (0, 255, 0): "small intestine",
    (0, 200, 100): "colon",
    (200, 150, 100): "abdominal wall",
    (250, 150, 100): "diaphragm",
    (255, 200, 100): "omentum",
    (180, 0, 0): "aorta",
    (0, 0, 180): "vena cava",
    (150, 100, 50): "liver",
    (0, 255, 255): "cystic duct",
    (0, 200, 255): "gallbladder",
    (0, 100, 255): "hepatic vein",
    (255, 150, 50): "hepatic ligament",
    (255, 220, 200): "cystic plate",
    (200, 100, 200): "stomach",
    (144, 238, 144): "common bile duct",
    (247, 255, 0): "mesenterium",
    (255, 206, 27): "hepatic duct",
    (200, 0, 200): "spleen",
    (255, 0, 150): "uterus",
    (255, 100, 200): "ovary",
    (200, 100, 255): "oviduct",
    (150, 0, 100): "prostate",
    (255, 200, 255): "urethra",
    (150, 100, 75): "ligated plexus",
    (200, 0, 150): "seminal vesicles",
    (100, 100, 100): "catheter",
    (255, 150, 255): "bladder",
    (100, 200, 255): "kidney",
    (150, 200, 255): "lung",
    (0, 150, 255): "airway",
    (255, 100, 100): "esophagus",
    (200, 200, 255): "pericardium",
    (100, 100, 255): "azygos vein",
    (0, 255, 150): "thoracic duct",
    (255, 255, 100): "nerves",
    (150, 150, 150): "ureter",
    (50, 50, 50): None,  # non-anatomical
    (0, 0, 0): None,  # background / excluded
    (173, 216, 230): "mesocolon",
    (255, 140, 0): "adrenal gland",
    (223, 3, 252): "pancreas",
    (0, 80, 100): "duodenum",
}

PROCEDURE_NAMES = {
    "adrenalectomy": "adrenalectomy",
    "appendectomy": "appendectomy",
    "cholecystectomy": "cholecystectomy",
    "colectomy": "colectomy",
    "esophagectomy": "esophagectomy",
    "gastric_surgery": "gastric surgery",
    "gastrojejunostomy": "gastrojejunostomy",
    "hemicolectomy": "hemicolectomy",
    "lar": "low anterior resection",
    "liver_resection": "liver resection",
    "rarp": "robot-assisted radical prostatectomy",
    "rectopexy": "rectopexy",
    "sigmoidcolectomy": "sigmoid colectomy",
    "splenectomy": "splenectomy",
}

# Minimum pixel count to consider a structure "visible" (0.5% of image)
MIN_PIXEL_FRACTION = 0.005


def extract_structures_from_mask(mask_path: str) -> list[str]:
    """Read a segmentation mask and return list of visible anatomical structures."""
    img = np.array(Image.open(mask_path).convert("RGB"))
    total_pixels = img.shape[0] * img.shape[1]
    min_pixels = int(total_pixels * MIN_PIXEL_FRACTION)

    # Reshape to (N, 3) and count unique colors
    pixels = img.reshape(-1, 3)
    # Use a fast approach: quantize to uint32 keys
    keys = pixels[:, 0].astype(np.uint32) * 65536 + pixels[:, 1].astype(np.uint32) * 256 + pixels[:, 2].astype(np.uint32)
    unique, counts = np.unique(keys, return_counts=True)

    structures = []
    for key, count in zip(unique, counts):
        if count < min_pixels:
            continue
        r = int(key // 65536)
        g = int((key % 65536) // 256)
        b = int(key % 256)
        color = (r, g, b)

        # Find closest color in palette (masks may have slight anti-aliasing)
        best_match = None
        best_dist = float("inf")
        for palette_color, name in COLOR_TO_STRUCTURE.items():
            dist = sum((a - b) ** 2 for a, b in zip(color, palette_color))
            if dist < best_dist:
                best_dist = dist
                best_match = name
        # Only accept matches within a reasonable distance
        if best_dist < 900 and best_match is not None:  # ~17 per channel tolerance
            structures.append(best_match)

    return sorted(set(structures))


def build_caption(procedure: str, structures: list[str]) -> str:
    """Build a natural language caption from procedure name and visible structures."""
    proc_name = PROCEDURE_NAMES.get(procedure, procedure.replace("_", " "))

    if not structures:
        return f"A laparoscopic view during {proc_name}"

    # Separate instruments from anatomy
    has_instruments = "surgical instruments" in structures
    anatomy = [s for s in structures if s != "surgical instruments"]

    parts = []
    if anatomy:
        if len(anatomy) <= 3:
            parts.append(", ".join(anatomy))
        else:
            parts.append(", ".join(anatomy[:3]) + f", and other structures")

    caption = f"A laparoscopic view during {proc_name}"
    if parts:
        caption += f" showing {parts[0]}"
    if has_instruments:
        caption += " with surgical instruments visible"

    return caption


def main():
    import sys
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/pkorzeniowsk/Datasets/atlas120k/train")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/pkorzeniowsk/Projects/nemo-automodel/data/atlas120k_flux")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    frame_count = 0

    procedures = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    for proc_dir in procedures:
        procedure = proc_dir.name
        print(f"\nProcessing {procedure}...")

        videos = sorted([d for d in proc_dir.iterdir() if d.is_dir()])
        for video_dir in videos:
            clips = sorted([d for d in video_dir.iterdir() if d.is_dir()])
            for clip_dir in clips:
                images_dir = clip_dir / "images"
                masks_dir = clip_dir / "masks"

                if not images_dir.exists():
                    continue

                frames = sorted(images_dir.glob("*.jpg"))
                if not frames:
                    continue

                # Take every SKIP-th frame
                selected = frames[::SKIP]

                for frame_path in selected:
                    frame_name = frame_path.stem  # e.g. frame_000328
                    mask_path = masks_dir / f"{frame_name}.png"

                    # Extract structures from mask if available
                    if mask_path.exists():
                        structures = extract_structures_from_mask(str(mask_path))
                    else:
                        structures = []

                    caption = build_caption(procedure, structures)

                    # Copy frame to output directory with unique name
                    out_name = f"{procedure}_{video_dir.name}_{clip_dir.name}_{frame_name}.jpg"
                    out_path = output_dir / out_name
                    shutil.copy2(frame_path, out_path)

                    metadata.append({
                        "file_name": out_name,
                        "text": caption,
                        "procedure": procedure,
                        "source_video": video_dir.name,
                        "source_clip": clip_dir.name,
                        "source_frame": frame_name,
                        "structures": structures,
                    })
                    frame_count += 1

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Extracted {frame_count} frames to {output_dir}")
    print(f"Metadata saved to {meta_path}")

    # Print stats
    proc_counts = Counter(m["procedure"] for m in metadata)
    print(f"\nFrames per procedure:")
    for proc, count in sorted(proc_counts.items(), key=lambda x: -x[1]):
        print(f"  {proc:25s} {count:5d}")

    # Sample captions
    print(f"\nSample captions:")
    import random
    random.seed(42)
    for m in random.sample(metadata, min(5, len(metadata))):
        print(f"  [{m['procedure']}] {m['text']}")


if __name__ == "__main__":
    main()
