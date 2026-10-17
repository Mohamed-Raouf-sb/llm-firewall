import argparse
import random
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a binary prompt risk classifier")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with train.csv and validation.csv")
    parser.add_argument("--output-dir", default="artifacts/prompt-firewall")
    parser.add_argument("--base-model", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--hard-negatives", default="data/raw/benign_hard_negatives.csv", help="Optional safe examples appended to training only; set to an empty string to disable")
    parser.add_argument("--hard-negative-repeat", type=int, default=5, help="Repeat the small hard-negative set during training to give it meaningful weight")
    parser.add_argument("--replay-per-class", type=int, default=0, help="When fine-tuning a saved model, replay this many examples per class from the original training split")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--no-checkpoints", action="store_true", help="Skip optimizer checkpoints to reduce disk usage; save the final model only")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    datasets = {}
    for split in ("train", "validation"):
        path = Path(args.data_dir) / f"{split}.csv"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Keep the dataset's train/validation/test splits in {args.data_dir}.")
        dataset = Dataset.from_csv(str(path))
        if not {"text", "label"}.issubset(dataset.column_names):
            raise SystemExit(f"{path} must contain text and label columns; found {dataset.column_names}")
        dataset = dataset.select_columns(["text", "label"])
        dataset = dataset.map(lambda row: {"label": int(row["label"])})
        labels = set(dataset["label"])
        if not labels.issubset({0, 1}) or labels != {0, 1}:
            raise SystemExit(f"{path} must contain both binary labels 0 (safe) and 1 (malicious); found {labels}")
        datasets[split] = dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length), batched=True)
    hard_repeated = None
    if args.hard_negatives:
        hard_path = Path(args.hard_negatives)
        if not hard_path.exists():
            raise SystemExit(f"Hard-negative file not found: {hard_path}")
        hard = Dataset.from_csv(str(hard_path)).select_columns(["text", "label"])
        if args.hard_negative_repeat < 1:
            raise SystemExit("--hard-negative-repeat must be at least 1")
        hard = hard.map(lambda row: {"label": int(row["label"])})
        if set(hard["label"]) != {0}:
            raise SystemExit(f"{hard_path} must contain safe label 0 only")
        existing = set(datasets["train"]["text"])
        hard = hard.filter(lambda row: row["text"] not in existing)
        hard = hard.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length), batched=True)
        hard_repeated = concatenate_datasets([hard] * args.hard_negative_repeat)
        datasets["train"] = concatenate_datasets([datasets["train"], hard_repeated])
        print(f"Added {len(hard)} benign hard negatives x {args.hard_negative_repeat}; validation and test are unchanged.")
    if args.replay_per_class:
        if args.replay_per_class < 1:
            raise SystemExit("--replay-per-class must be positive")
        original = Dataset.from_csv(str(Path(args.data_dir) / "train.csv")).select_columns(["text", "label"])
        replay_indices = []
        rng = random.Random(42)
        for target in (0, 1):
            candidates = [i for i, label in enumerate(original["label"]) if int(label) == target]
            if len(candidates) < args.replay_per_class:
                raise SystemExit(f"Not enough original training examples for class {target}")
            replay_indices.extend(rng.sample(candidates, args.replay_per_class))
        replay = original.select(replay_indices)
        replay = replay.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length), batched=True)
        datasets["train"] = concatenate_datasets([replay, hard_repeated]) if hard_repeated is not None else replay
        print(f"Using balanced replay of {args.replay_per_class} original examples per class.")
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2, id2label={0: "safe", 1: "malicious"}, label2id={"safe": 0, "malicious": 1})

    def metrics(prediction):
        import numpy as np
        pred = np.argmax(prediction.predictions, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(prediction.label_ids, pred, average="binary", zero_division=0)
        return {"accuracy": accuracy_score(prediction.label_ids, pred), "precision": precision, "recall": recall, "f1": f1}

    training = TrainingArguments(output_dir=args.output_dir, learning_rate=args.learning_rate, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size, eval_strategy="epoch", save_strategy="no" if args.no_checkpoints else "epoch", load_best_model_at_end=not args.no_checkpoints, metric_for_best_model="f1", greater_is_better=True, report_to="none", seed=42, data_seed=42)
    trainer = Trainer(model=model, args=training, train_dataset=datasets["train"], eval_dataset=datasets["validation"], processing_class=tokenizer, compute_metrics=metrics)
    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Validation metrics: {validation_metrics}")


if __name__ == "__main__":
    main()
