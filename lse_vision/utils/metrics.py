from __future__ import annotations

import csv
import json
from pathlib import Path

def build_classification_metrics(targets: list[int], predictions: list[int], class_names: list[str]) -> dict:
    num_classes = len(class_names)
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for actual, predicted in zip(targets, predictions):
        matrix[actual][predicted] += 1

    correct = sum(matrix[index][index] for index in range(num_classes))
    total = max(1, len(targets))
    accuracy = correct / total

    per_class_report: dict[str, dict[str, float]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    for class_index, class_name in enumerate(class_names):
        true_positive = matrix[class_index][class_index]
        false_positive = sum(matrix[row][class_index] for row in range(num_classes) if row != class_index)
        false_negative = sum(matrix[class_index][column] for column in range(num_classes) if column != class_index)
        support = sum(matrix[class_index])

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1_score)
        per_class_report[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
            "support": float(support),
        }

    macro_precision = sum(precision_values) / num_classes
    macro_recall = sum(recall_values) / num_classes
    macro_f1 = sum(f1_values) / num_classes

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": matrix,
        "classification_report": per_class_report,
    }


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_confusion_matrix_csv(matrix: list[list[int]], class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])
