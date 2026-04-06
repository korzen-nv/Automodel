#!/usr/bin/env python3
"""Prepare HeiChole surgical frames for FLUX fine-tuning with nemo-automodel.

Reads frames from the HeiSurf/Frames directory, maps each to phase/action/instrument
annotations, filters censored (all-white) frames, and writes image + sidecar .txt
caption files to an output directory.
"""

import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/pkorzeniowsk/Datasets/HeiSurf")
FRAMES_DIR = DATASET_ROOT / "Frames"
PHASE_DIR = DATASET_ROOT / "Phase"
ACTION_DIR = DATASET_ROOT / "Action"
INSTRUMENT_DIR = DATASET_ROOT / "Instrument"
OUTPUT_DIR = Path("/home/pkorzeniowsk/Projects/nemo-automodel/data/heichole_flux")

# ── Annotation maps ─────────────────────────────────────────────────────────
PHASE_NAMES = {
    0: "Preparation",
    1: "Calot triangle dissection",
    2: "Clipping and cutting",
    3: "Gallbladder dissection",
    4: "Gallbladder packaging",
    5: "Cleaning and coagulation",
    6: "Gallbladder retraction",
}

ACTION_NAMES = {
    0: "Grasp",
    1: "Hold",
    2: "Cut",
    3: "Clip",
}

TOOL_CATEGORY_NAMES = {
    0: "Grasper",
    1: "Clipper",
    2: "Coagulation instrument",
    3: "Scissors",
    4: "Suction-irrigation",
    5: "Specimen bag",
    6: "Stapler",
    20: "Undefined instrument shaft",
}

TOOL_NAMES = {
    0: "Curved atraumatic grasper",
    1: "Toothed grasper",
    2: "Fenestrated toothed grasper",
    3: "Atraumatic grasper",
    4: "Overholt",
    5: "LigaSure",
    6: "Electric hook",
    7: "Scissors",
    8: "Clip-applier (metal)",
    9: "Clip-applier (Hem-O-Lok)",
    10: "Swab grasper",
    11: "Argon beamer",
    12: "Suction-irrigation",
    13: "Specimen bag",
    14: "Tiger mouth forceps",
    15: "Claw forceps",
    16: "Atraumatic grasper (short)",
    17: "Crocodile forceps",
    18: "Flat grasper",
    19: "Pointed forceps",
    20: "Stapler",
    30: "Undefined instrument shaft",
}

# Phase-specific descriptive context
PHASE_CONTEXT = {
    0: "The surgical team is preparing the operative field and positioning instruments for the laparoscopic procedure.",
    1: "The surgeon is dissecting the Calot triangle to identify and isolate the cystic duct and cystic artery.",
    2: "The surgeon is applying clips to the cystic duct and cystic artery, then cutting between the clips.",
    3: "The surgeon is dissecting the gallbladder from the liver bed using electrocautery.",
    4: "The gallbladder is being placed into a specimen bag for extraction.",
    5: "The surgical field is being cleaned and any bleeding points are being coagulated.",
    6: "The gallbladder is being retracted and extracted from the abdominal cavity.",
}

FPS = 25  # Video framerate


def frame_filename_to_annotation_index(frame_number: int) -> int:
    """Convert frame filename number to annotation row index.

    Frames are sampled at 2-minute intervals starting at 30s.
    frame0000 → 30s, frame0002 → 2m30s, frame0004 → 4m30s, etc.
    """
    timestamp_seconds = (frame_number // 2) * 120 + 30
    return timestamp_seconds * FPS


def load_csv_annotations(filepath: Path) -> dict[int, list[int]]:
    """Load a CSV annotation file. Returns {frame_idx: [values...]}."""
    annotations = {}
    with open(filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            frame_idx = int(row[0])
            values = [int(x) for x in row[1:]]
            annotations[frame_idx] = values
    return annotations


def is_censored(image_path: Path, white_threshold: float = 0.98) -> bool:
    """Check if a frame is censored (nearly all-white)."""
    img = np.array(Image.open(image_path))
    white_pixels = np.all(img > 250, axis=-1)
    return white_pixels.mean() > white_threshold


def build_caption(
    phase_id: int,
    actions: list[int],
    instruments: list[int],
) -> str:
    """Build a rich text caption from annotations."""
    parts = []

    # Base description
    parts.append("Endoscopic view during laparoscopic cholecystectomy (gallbladder removal surgery).")

    # Phase
    phase_name = PHASE_NAMES.get(phase_id, "Unknown phase")
    parts.append(f"Current surgical phase: {phase_name}.")

    # Phase context
    context = PHASE_CONTEXT.get(phase_id)
    if context:
        parts.append(context)

    # Actions
    active_actions = [ACTION_NAMES[i] for i, v in enumerate(actions) if v == 1]
    if active_actions:
        parts.append(f"Surgical actions being performed: {', '.join(active_actions).lower()}.")

    # Instruments (using the 21-column instrument annotation: columns map to tool IDs)
    # The instrument CSV has 21 columns after frame_idx, mapping to tool IDs 0-20
    # (with gaps — tool IDs 0-20 as listed, plus tool 30 mapped to column 21)
    active_tools = []
    for col_idx, present in enumerate(instruments):
        if present == 1 and col_idx in TOOL_NAMES:
            active_tools.append(TOOL_NAMES[col_idx])
    if active_tools:
        parts.append(f"Visible instruments: {', '.join(active_tools).lower()}.")
    else:
        parts.append("No surgical instruments currently visible in the field of view.")

    return " ".join(parts)


def process_dataset():
    """Main processing pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_frames": 0,
        "censored": 0,
        "missing_annotations": 0,
        "written": 0,
    }

    for video_num in range(1, 25):
        video_dir = FRAMES_DIR / f"{video_num:02d}"
        if not video_dir.exists():
            print(f"  Skipping video {video_num:02d}: directory not found")
            continue

        # Load annotations
        phase_file = PHASE_DIR / f"hei-chole{video_num}_annotation_phase.csv"
        action_file = ACTION_DIR / f"hei-chole{video_num}_annotation_action.csv"
        instrument_file = INSTRUMENT_DIR / f"hei-chole{video_num}_annotation_instrument.csv"

        if not all(f.exists() for f in [phase_file, action_file, instrument_file]):
            print(f"  Skipping video {video_num:02d}: missing annotation files")
            continue

        phase_annot = load_csv_annotations(phase_file)
        action_annot = load_csv_annotations(action_file)
        instrument_annot = load_csv_annotations(instrument_file)

        max_annot_frame = max(phase_annot.keys()) if phase_annot else 0

        # Process each frame
        frame_files = sorted(video_dir.glob("hei-chole_*.png"))
        for frame_path in frame_files:
            stats["total_frames"] += 1

            # Extract frame number from filename: hei-chole_01_frame0036.png → 36
            fname = frame_path.stem  # hei-chole_01_frame0036
            frame_number = int(fname.split("frame")[-1])

            # Map to annotation index
            annot_idx = frame_filename_to_annotation_index(frame_number)

            # Clamp to valid range
            annot_idx = min(annot_idx, max_annot_frame)

            # Check if censored
            if is_censored(frame_path):
                stats["censored"] += 1
                continue

            # Get annotations (use closest available if exact index missing)
            phase_vals = phase_annot.get(annot_idx)
            action_vals = action_annot.get(annot_idx)
            instrument_vals = instrument_annot.get(annot_idx)

            if phase_vals is None or action_vals is None or instrument_vals is None:
                # Try nearby frames (within 50 frames = 2 seconds)
                for offset in range(1, 51):
                    for direction in [1, -1]:
                        nearby = annot_idx + offset * direction
                        if phase_vals is None and nearby in phase_annot:
                            phase_vals = phase_annot[nearby]
                        if action_vals is None and nearby in action_annot:
                            action_vals = action_annot[nearby]
                        if instrument_vals is None and nearby in instrument_annot:
                            instrument_vals = instrument_annot[nearby]
                    if all(v is not None for v in [phase_vals, action_vals, instrument_vals]):
                        break

            if phase_vals is None:
                stats["missing_annotations"] += 1
                print(f"  Warning: no annotations for video {video_num:02d} frame {frame_number} (annot_idx={annot_idx})")
                continue

            # Build caption
            phase_id = phase_vals[0]
            caption = build_caption(phase_id, action_vals or [0, 0, 0, 0], instrument_vals or [])

            # Write output
            out_name = f"heichole{video_num:02d}_frame{frame_number:04d}"
            out_img = OUTPUT_DIR / f"{out_name}.png"
            out_txt = OUTPUT_DIR / f"{out_name}.txt"

            shutil.copy2(frame_path, out_img)
            out_txt.write_text(caption)
            stats["written"] += 1

        print(f"  Video {video_num:02d}: processed {len(frame_files)} frames")

    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"  Total frames scanned: {stats['total_frames']}")
    print(f"  Censored (skipped):   {stats['censored']}")
    print(f"  Missing annotations:  {stats['missing_annotations']}")
    print(f"  Written to output:    {stats['written']}")
    print(f"  Output directory:     {OUTPUT_DIR}")


if __name__ == "__main__":
    process_dataset()
