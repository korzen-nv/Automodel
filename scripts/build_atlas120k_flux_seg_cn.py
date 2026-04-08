#!/usr/bin/env python3
"""Build Atlas120k manifests and control images for FLUX segmentation ControlNet."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

CLASS_TO_COLOR = {
    1: (255, 255, 255),
    2: (0, 0, 255),
    3: (255, 0, 0),
    4: (255, 255, 0),
    5: (0, 255, 0),
    6: (0, 200, 100),
    7: (200, 150, 100),
    8: (250, 150, 100),
    9: (255, 200, 100),
    10: (180, 0, 0),
    11: (0, 0, 180),
    12: (150, 100, 50),
    13: (0, 255, 255),
    14: (0, 200, 255),
    15: (0, 100, 255),
    16: (255, 150, 50),
    17: (255, 220, 200),
    18: (200, 100, 200),
    19: (144, 238, 144),
    20: (247, 255, 0),
    21: (255, 206, 27),
    22: (200, 0, 200),
    23: (255, 0, 150),
    24: (255, 100, 200),
    25: (200, 100, 255),
    26: (150, 0, 100),
    27: (255, 200, 255),
    28: (150, 100, 75),
    29: (200, 0, 150),
    30: (100, 100, 100),
    31: (255, 150, 255),
    32: (100, 200, 255),
    33: (150, 200, 255),
    34: (0, 150, 255),
    35: (255, 100, 100),
    36: (200, 200, 255),
    37: (100, 100, 255),
    38: (0, 255, 150),
    39: (255, 255, 100),
    40: (150, 150, 150),
    41: (50, 50, 50),
    42: (0, 0, 0),
    43: (173, 216, 230),
    44: (255, 140, 0),
    45: (223, 3, 252),
    46: (0, 80, 100),
}

CLASS_TO_STRUCTURE = {
    1: "surgical instruments",
    2: "major vein",
    3: "major artery",
    4: "major nerve",
    5: "small intestine",
    6: "colon",
    7: "abdominal wall",
    8: "diaphragm",
    9: "omentum",
    10: "aorta",
    11: "vena cava",
    12: "liver",
    13: "cystic duct",
    14: "gallbladder",
    15: "hepatic vein",
    16: "hepatic ligament",
    17: "cystic plate",
    18: "stomach",
    19: "common bile duct",
    20: "mesenterium",
    21: "hepatic duct",
    22: "spleen",
    23: "uterus",
    24: "ovary",
    25: "oviduct",
    26: "prostate",
    27: "urethra",
    28: "ligated plexus",
    29: "seminal vesicles",
    30: "catheter",
    31: "bladder",
    32: "kidney",
    33: "lung",
    34: "airway",
    35: "esophagus",
    36: "pericardium",
    37: "azygos vein",
    38: "thoracic duct",
    39: "nerves",
    40: "ureter",
    41: None,
    42: None,
    43: "mesocolon",
    44: "adrenal gland",
    45: "pancreas",
    46: "duodenum",
}

CHOLE_CLASS_IDS = {1, 3, 5, 7, 9, 12, 13, 14, 16, 17, 18, 42}
CLASS_PRESETS = {
    "full": set(CLASS_TO_COLOR),
    "chole": CHOLE_CLASS_IDS,
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

IGNORE_CLASS_IDS = {41, 42}
BLACK = (0, 0, 0)
MAX_COLOR_DIST2 = 900

PALETTE_CLASS_IDS = np.array(sorted(CLASS_TO_COLOR), dtype=np.int32)
PALETTE_COLORS = np.array([CLASS_TO_COLOR[class_id] for class_id in PALETTE_CLASS_IDS], dtype=np.int16)
EXACT_KEY_TO_CLASS = {
    ((rgb[0] << 16) | (rgb[1] << 8) | rgb[2]): class_id for class_id, rgb in CLASS_TO_COLOR.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--include_splits", nargs="+", default=None)
    parser.add_argument("--procedure_filter", nargs="+", default=None)
    parser.add_argument("--class_preset", choices=sorted(CLASS_PRESETS), default="full")
    parser.add_argument("--train_on_all_rows", action="store_true")
    parser.add_argument("--fixed_caption", type=str, default=None)
    parser.add_argument("--conditioning_format", choices=["rgb_png", "onehot_npy"], default="rgb_png")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--val_clip_fraction", type=float, default=0.03)
    parser.add_argument("--eval_examples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_pixel_fraction", type=float, default=0.005)
    parser.add_argument("--limit_clips", type=int, default=None)
    return parser.parse_args()


def rgb_to_keys(pixels: np.ndarray) -> np.ndarray:
    return (
        pixels[:, 0].astype(np.uint32) * 65536
        + pixels[:, 1].astype(np.uint32) * 256
        + pixels[:, 2].astype(np.uint32)
    )


def key_to_rgb(key: int) -> tuple[int, int, int]:
    return int(key // 65536), int((key % 65536) // 256), int(key % 256)


def classify_color(key: int) -> tuple[int | None, int]:
    class_id = EXACT_KEY_TO_CLASS.get(key)
    if class_id is not None:
        return class_id, 0

    color = np.array(key_to_rgb(key), dtype=np.int32)
    distances = ((PALETTE_COLORS.astype(np.int32) - color) ** 2).sum(axis=1)
    palette_index = int(distances.argmin())
    best_dist = int(distances[palette_index])
    if best_dist > MAX_COLOR_DIST2:
        return None, best_dist
    return int(PALETTE_CLASS_IDS[palette_index]), best_dist


def normalize_mask(
    mask: np.ndarray, min_pixels: int, allowed_class_ids: set[int], active_class_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    flat = mask.reshape(-1, 3)
    keys = rgb_to_keys(flat)
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)

    mapped_colors = np.zeros((len(unique_keys), 3), dtype=np.uint8)
    mapped_channels = np.full((len(unique_keys),), -1, dtype=np.int16)
    class_pixel_counts: Counter[int] = Counter()
    unknown_pixels = 0
    unknown_colors = 0
    class_to_channel = {class_id: idx for idx, class_id in enumerate(active_class_ids)}

    for idx, (key, count) in enumerate(zip(unique_keys.tolist(), counts.tolist(), strict=False)):
        class_id, best_dist = classify_color(int(key))
        if class_id is None:
            unknown_pixels += int(count)
            unknown_colors += 1
            mapped_colors[idx] = BLACK
            continue
        if class_id not in allowed_class_ids or class_id in IGNORE_CLASS_IDS:
            color = BLACK
        else:
            color = CLASS_TO_COLOR[class_id]
            mapped_channels[idx] = class_to_channel[class_id]
        mapped_colors[idx] = np.array(color, dtype=np.uint8)
        if class_id in allowed_class_ids and class_id not in IGNORE_CLASS_IDS:
            class_pixel_counts[class_id] += int(count)

    normalized = mapped_colors[inverse].reshape(mask.shape)
    channel_map = mapped_channels[inverse].reshape(mask.shape[:2])
    visible_structures = sorted(
        {
            CLASS_TO_STRUCTURE[class_id]
            for class_id, count in class_pixel_counts.items()
            if count >= min_pixels and CLASS_TO_STRUCTURE[class_id] is not None
        }
    )

    stats = {
        "foreground_pixels": int(sum(class_pixel_counts.values())),
        "unknown_pixels": int(unknown_pixels),
        "unknown_colors": int(unknown_colors),
    }
    return normalized, channel_map, visible_structures, stats


def channel_map_to_onehot(channel_map: np.ndarray, num_channels: int) -> np.ndarray:
    onehot = np.zeros((num_channels, channel_map.shape[0], channel_map.shape[1]), dtype=np.uint8)
    for channel_idx in range(num_channels):
        onehot[channel_idx] = (channel_map == channel_idx).astype(np.uint8)
    return onehot


def build_caption(procedure: str, structures: list[str]) -> str:
    proc_name = PROCEDURE_NAMES.get(procedure, procedure.replace("_", " "))
    if not structures:
        return f"A laparoscopic view during {proc_name}"

    has_instruments = "surgical instruments" in structures
    anatomy = [structure for structure in structures if structure != "surgical instruments"]

    caption = f"A laparoscopic view during {proc_name}"
    if anatomy:
        shown = ", ".join(anatomy[:3])
        if len(anatomy) > 3:
            shown += ", and other structures"
        caption += f" showing {shown}"
    if has_instruments:
        caption += " with surgical instruments visible"
    return caption


def resolve_source_roots(dataset_root: Path, include_splits: list[str] | None) -> list[tuple[str, Path]]:
    if include_splits:
        return [(split, (dataset_root / split).resolve()) for split in include_splits]

    split_name = dataset_root.name if dataset_root.name in {"train", "val", "test"} else "source"
    return [(split_name, dataset_root)]


def clip_split(clip_id: str, val_clip_fraction: float) -> str:
    digest = hashlib.md5(clip_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_clip_fraction else "train"


def choose_eval_rows(rows: list[dict], eval_examples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["procedure"]].append(row)

    for procedure_rows in grouped.values():
        rng.shuffle(procedure_rows)

    chosen: list[dict] = []
    procedures = sorted(grouped)
    while len(chosen) < eval_examples and any(grouped.values()):
        for procedure in procedures:
            if grouped[procedure] and len(chosen) < eval_examples:
                chosen.append(grouped[procedure].pop())
    return chosen


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    allowed_class_ids = CLASS_PRESETS[args.class_preset]
    procedure_filter = set(args.procedure_filter or [])
    source_roots = resolve_source_roots(dataset_root, args.include_splits)
    active_class_ids = sorted(class_id for class_id in allowed_class_ids if class_id not in IGNORE_CLASS_IDS)
    manifests_dir = output_dir / "manifests"
    control_root = output_dir / "control_pngs"
    onehot_root = output_dir / "control_onehot"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    control_root.mkdir(parents=True, exist_ok=True)
    if args.conditioning_format == "onehot_npy":
        onehot_root.mkdir(parents=True, exist_ok=True)

    train_path = manifests_dir / "train.jsonl"
    val_path = manifests_dir / "val.jsonl"

    min_pixels_cache: dict[tuple[int, int], int] = {}
    stats: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "include_splits": [name for name, _ in source_roots],
        "procedure_filter": sorted(procedure_filter),
        "class_preset": args.class_preset,
        "train_on_all_rows": args.train_on_all_rows,
        "fixed_caption": args.fixed_caption,
        "conditioning_format": args.conditioning_format,
        "active_class_ids": active_class_ids,
        "frame_stride": args.frame_stride,
        "val_clip_fraction": args.val_clip_fraction,
        "split_counts": Counter(),
        "source_split_counts": Counter(),
        "procedure_counts": Counter(),
        "structure_counts": Counter(),
        "skipped_missing_masks": 0,
        "skipped_empty_masks": 0,
        "unknown_pixels": 0,
        "unknown_colors": 0,
    }
    val_rows: list[dict] = []
    eval_candidate_rows: list[dict] = []

    clip_entries: list[tuple[str, Path]] = []
    for source_split, source_root in source_roots:
        for clip_dir in sorted(source_root.glob("*/*/*")):
            clip_entries.append((source_split, clip_dir))
    if args.limit_clips is not None:
        clip_entries = clip_entries[: args.limit_clips]

    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open("w", encoding="utf-8") as val_handle:
        for source_split, clip_dir in tqdm(clip_entries, desc="Atlas clips"):
            if not clip_dir.is_dir():
                continue
            images_dir = clip_dir / "images"
            masks_dir = clip_dir / "masks"
            if not images_dir.is_dir() or not masks_dir.is_dir():
                continue

            procedure = clip_dir.parts[-3]
            if procedure_filter and procedure not in procedure_filter:
                continue
            video = clip_dir.parts[-2]
            clip = clip_dir.parts[-1]
            clip_id = f"{source_split}/{procedure}/{video}/{clip}"
            split = "train" if args.train_on_all_rows else clip_split(clip_id, args.val_clip_fraction)

            frame_paths = sorted(images_dir.glob("*.jpg"))
            for frame_index, frame_path in enumerate(frame_paths):
                if frame_index % args.frame_stride != 0:
                    continue

                mask_path = masks_dir / f"{frame_path.stem}.png"
                if not mask_path.exists():
                    stats["skipped_missing_masks"] += 1
                    continue

                mask = np.array(Image.open(mask_path).convert("RGB"))
                shape_key = (mask.shape[0], mask.shape[1])
                if shape_key not in min_pixels_cache:
                    min_pixels_cache[shape_key] = int(mask.shape[0] * mask.shape[1] * args.min_pixel_fraction)
                min_pixels = min_pixels_cache[shape_key]

                normalized_mask, channel_map, structures, mask_stats = normalize_mask(
                    mask, min_pixels, allowed_class_ids, active_class_ids
                )
                if mask_stats["foreground_pixels"] == 0:
                    stats["skipped_empty_masks"] += 1
                    continue

                preview_path = control_root / source_split / procedure / video / clip / f"{frame_path.stem}.png"
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(normalized_mask, mode="RGB").save(preview_path, compress_level=0)

                conditioning_path: Path
                if args.conditioning_format == "onehot_npy":
                    conditioning_path = onehot_root / source_split / procedure / video / clip / f"{frame_path.stem}.npy"
                    conditioning_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(conditioning_path, channel_map_to_onehot(channel_map, len(active_class_ids)))
                else:
                    conditioning_path = preview_path

                caption = args.fixed_caption or build_caption(procedure, structures)
                row = {
                    "image": str(frame_path.resolve()),
                    "conditioning_image": str(conditioning_path),
                    "conditioning_preview": str(preview_path),
                    "text": caption,
                    "procedure": procedure,
                    "source_split": source_split,
                    "clip_id": clip_id,
                    "frame_name": frame_path.name,
                    "structures": structures,
                }

                target_handle = train_handle if split == "train" else val_handle
                target_handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                if split == "val":
                    val_rows.append(row)
                eval_candidate_rows.append(row)

                stats["split_counts"][split] += 1
                stats["source_split_counts"][source_split] += 1
                stats["procedure_counts"][procedure] += 1
                stats["structure_counts"].update(structures)
                stats["unknown_pixels"] += mask_stats["unknown_pixels"]
                stats["unknown_colors"] += mask_stats["unknown_colors"]

    if args.train_on_all_rows:
        eval_rows = choose_eval_rows(eval_candidate_rows, args.eval_examples, args.seed)
        write_jsonl(val_path, eval_rows)
    else:
        eval_rows = choose_eval_rows(val_rows, args.eval_examples, args.seed)
    eval_path = manifests_dir / "val_eval.jsonl"
    write_jsonl(eval_path, eval_rows)

    serializable_stats = {
        "dataset_root": stats["dataset_root"],
        "output_dir": stats["output_dir"],
        "include_splits": stats["include_splits"],
        "procedure_filter": stats["procedure_filter"],
        "class_preset": stats["class_preset"],
        "train_on_all_rows": stats["train_on_all_rows"],
        "fixed_caption": stats["fixed_caption"],
        "conditioning_format": stats["conditioning_format"],
        "active_class_ids": stats["active_class_ids"],
        "frame_stride": stats["frame_stride"],
        "val_clip_fraction": stats["val_clip_fraction"],
        "split_counts": dict(stats["split_counts"]),
        "source_split_counts": dict(stats["source_split_counts"]),
        "procedure_counts": dict(stats["procedure_counts"]),
        "structure_counts": dict(stats["structure_counts"]),
        "skipped_missing_masks": stats["skipped_missing_masks"],
        "skipped_empty_masks": stats["skipped_empty_masks"],
        "unknown_pixels": stats["unknown_pixels"],
        "unknown_colors": stats["unknown_colors"],
        "eval_examples": len(eval_rows),
    }
    (output_dir / "stats.json").write_text(json.dumps(serializable_stats, indent=2), encoding="utf-8")
    (output_dir / "class_ids.json").write_text(json.dumps(active_class_ids, indent=2), encoding="utf-8")

    print(json.dumps(serializable_stats, indent=2))
    print(f"Train manifest: {train_path}")
    print(f"Val manifest:   {val_path}")
    print(f"Eval manifest:  {eval_path}")


if __name__ == "__main__":
    main()
