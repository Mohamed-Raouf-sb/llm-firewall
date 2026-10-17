import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LABELS = {"0": 0, "safe": 0, "benign": 0, "1": 1, "malicious": 1, "injection": 1, "jailbreak": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and split a text,label CSV")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if not {"text", "label"}.issubset(frame.columns):
        raise SystemExit("Input CSV must contain text and label columns")
    frame = frame[["text", "label"]].dropna()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower().map(LABELS)
    frame = frame.dropna().drop_duplicates(subset=["text"])
    frame["label"] = frame["label"].astype(int)
    if frame["label"].nunique() != 2 or frame["label"].value_counts().min() < 3:
        raise SystemExit("Need at least three examples from each class for stratified train/validation/test splits")
    train, rest = train_test_split(frame, test_size=0.2, random_state=args.seed, stratify=frame["label"])
    valid, test = train_test_split(rest, test_size=0.5, random_state=args.seed, stratify=rest["label"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("validation", valid), ("test", test)):
        part.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows; classes={part.label.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
