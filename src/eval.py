from datasets import Dataset
from os import path
from pandas import DataFrame
from transformers import Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from __params__ import RESULTS_PATH


def evaluate(trainer: Trainer, test: Dataset) -> DataFrame:
    FILE = path.join(RESULTS_PATH, "evaluation.csv")

    results = trainer.evaluate(eval_dataset=test)

    df = DataFrame({
        "model": [trainer.model.__class__.__name__],
        "accuracy": results["eval_accuracy"],
        "precision-": results["eval_precision-"],
        "precision~": results["eval_precision~"],
        "precision+": results["eval_precision+"],
        "precision": results["eval_precision"],
        "recall-": results["eval_recall-"],
        "recall~": results["eval_recall~"],
        "recall+": results["eval_recall+"],
        "recall": results["eval_recall"],
        "f1-": results["eval_f1-"],
        "f1~": results["eval_f1~"],
        "f1+": results["eval_f1+"],
        "f1": results["eval_f1"],
        "metadata": [str(trainer.args)]
    }).to_csv(FILE, index=False)

    print(f"Results saved to {FILE}.")

    return df


def compute_metrics(p) -> dict[str, float]:
    preds, labels = p.predictions.argmax(-1), p.label_ids

    accuracy = accuracy_score(labels, preds)
    precisions = precision_score(labels, preds, average=None)
    recalls = recall_score(labels, preds, average=None)
    f1s = f1_score(labels, preds, average=None)

    return {
        "accuracy": accuracy,
        "precision-": precisions[0],
        "precision~": precisions[1],
        "precision+": precisions[2],
        "precision": precisions.mean(),
        "recall-": recalls[0],
        "recall~": recalls[1],
        "recall+": recalls[2],
        "recall": recalls.mean(),
        "f1-": f1s[0],
        "f1~": f1s[1],
        "f1+": f1s[2],
        "f1": f1s.mean()
    }
