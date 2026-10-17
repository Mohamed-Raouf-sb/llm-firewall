"""Download Neuralchemy's core dataset and export project-compatible CSV splits.

Requires the `datasets` package and an internet connection to Hugging Face.
The resulting train/validation/test files can be passed directly to train.py.
"""
import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    dataset = load_dataset(
        "neuralchemy/Prompt-injection-dataset",
        "core",
        cache_dir=args.cache_dir,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        rows = dataset[split]
        destination = output_dir / f"{split}.csv"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("text", "label"))
            writer.writeheader()
            for row in rows:
                writer.writerow({"text": row["text"], "label": int(row["label"])})
        print(f"Wrote {len(rows)} rows to {destination}")


if __name__ == "__main__":
    main()
