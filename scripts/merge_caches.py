"""Merge multiple FLUX preprocessing caches into a single virtual cache directory."""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: merge_caches.py <output_dir> <cache_dir1> <cache_dir2> [...]")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    cache_dirs = [Path(d) for d in sys.argv[2:]]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_items = []
    total_from = {}

    for cache_dir in cache_dirs:
        meta_file = cache_dir / "metadata.json"
        if not meta_file.exists():
            print(f"WARNING: No metadata.json in {cache_dir}, skipping")
            continue

        meta = json.loads(meta_file.read_text())
        count = 0
        for shard_name in meta["shards"]:
            shard_path = cache_dir / shard_name
            items = json.loads(shard_path.read_text())
            all_items.extend(items)
            count += len(items)

        total_from[str(cache_dir)] = count
        print(f"  {cache_dir.name}: {count} items")

    # Write merged shard
    shard_name = "metadata_shard_0000.json"
    with open(output_dir / shard_name, "w") as f:
        json.dump(all_items, f)

    # Write merged metadata
    merged_meta = {
        "processor": "flux",
        "model_name": "black-forest-labs/FLUX.1-dev",
        "model_type": "flux",
        "total_items": len(all_items),
        "num_shards": 1,
        "shard_size": len(all_items),
        "shards": [shard_name],
        "caption_field": "internvl",
        "max_pixels": 262144,
        "sources": total_from,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(merged_meta, f, indent=2)

    print(f"\nMerged {len(all_items)} total items into {output_dir}")
    for src, count in total_from.items():
        print(f"  {Path(src).name}: {count} ({count*100//len(all_items)}%)")


if __name__ == "__main__":
    main()
