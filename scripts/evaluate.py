import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved classifier on labeled CSV data")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--tune-threshold", action="store_true", help="Choose a threshold on this validation set to minimize FPR while targeting recall")
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--output", default=None, help="Optional JSON path for the metrics report")
    args = parser.parse_args()
    frame = pd.read_csv(args.data).dropna(subset=["text", "label"])
    if not {"text", "label"}.issubset(frame.columns):
        raise SystemExit("Evaluation CSV must contain text and label columns")
    frame = frame[["text", "label"]].copy()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["label"])
    truth = frame["label"].astype(int).tolist()
    if not set(truth).issubset({0, 1}) or set(truth) != {0, 1}:
        raise SystemExit("Evaluation labels must include both 0 (safe) and 1 (malicious)")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = Dataset.from_dict({"text": frame["text"].tolist(), "label": truth})
    dataset = dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=512), batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir="/tmp/firewall-eval", per_device_eval_batch_size=32, report_to="none"), processing_class=tokenizer)
    logits = trainer.predict(dataset).predictions
    import numpy as np
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    class_one_id = int(model.config.label2id.get("malicious", 1))
    scores = probabilities[:, class_one_id]
    if args.tune_threshold:
        if not 0 < args.target_recall <= 1:
            raise SystemExit("--target-recall must be in (0, 1]")
        candidates = sorted(set(float(score) for score in scores), reverse=True)
        feasible = []
        for threshold in candidates:
            pred = [int(score >= threshold) for score in scores]
            tp = sum(p == 1 and y == 1 for p, y in zip(pred, truth))
            recall = tp / max(sum(y == 1 for y in truth), 1)
            if recall >= args.target_recall:
                tn = sum(p == 0 and y == 0 for p, y in zip(pred, truth))
                fp = sum(p == 1 and y == 0 for p, y in zip(pred, truth))
                feasible.append((fp / max(fp + tn, 1), -threshold, threshold))
        if not feasible:
            raise SystemExit("No threshold met target recall")
        args.threshold = min(feasible)[2]
    predicted = [int(score >= args.threshold) for score in scores]
    p, r, f, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    report = {"data": args.data, "model": args.model, "threshold": args.threshold, "accuracy": accuracy_score(truth, predicted), "precision": p, "recall": r, "f1": f, "false_positive_rate": fp / max(fp + tn, 1), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
