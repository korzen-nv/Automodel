#!/usr/bin/env python3
"""Convert .txt sidecar captions to JSONL format expected by nemo-automodel image preprocessing."""

import json
from pathlib import Path

DATA_DIR = Path("/home/pkorzeniowsk/Projects/nemo-automodel/data/heichole_flux")

count = 0
for txt_file in sorted(DATA_DIR.glob("*.txt")):
    image_name = txt_file.stem + ".png"
    image_path = DATA_DIR / image_name
    if not image_path.exists():
        continue

    caption = txt_file.read_text().strip()

    # The JSONLCaptionLoader looks for {stem}_internvl.json for each image
    jsonl_path = DATA_DIR / f"{txt_file.stem}_internvl.json"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps({"file_name": image_name, "internvl": caption}) + "\n")
    count += 1

print(f"Created {count} JSONL caption files in {DATA_DIR}")
