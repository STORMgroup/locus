"""Classification metrics of the benchmark tables (Matthews correlation, accuracy, F1, ROC AUC)."""
import math
import statistics

import numpy as np


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def matthews_correlation(matrix: np.ndarray) -> float:
    samples = float(matrix.sum())
    correct = float(np.trace(matrix))
    predicted = matrix.sum(axis=0).astype(np.float64)
    observed = matrix.sum(axis=1).astype(np.float64)
    numerator = correct * samples - float(np.dot(predicted, observed))
    denominator = math.sqrt(
        max(samples * samples - float(np.dot(predicted, predicted)), 0.0)
        * max(samples * samples - float(np.dot(observed, observed)), 0.0)
    )
    return numerator / denominator if denominator > 0.0 else 0.0


def macro_f1(matrix: np.ndarray) -> float:
    values = []
    for index in range(matrix.shape[0]):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - matrix[index, index])
        false_negative = float(matrix[index, :].sum() - matrix[index, index])
        denominator = 2.0 * true_positive + false_positive + false_negative
        values.append(2.0 * true_positive / denominator if denominator else 0.0)
    return statistics.fmean(values)


def positive_class_f1(matrix: np.ndarray) -> float:
    if matrix.shape != (2, 2):
        raise ValueError("Positive-class F1 is defined here only for binary tasks")
    true_positive = float(matrix[1, 1])
    false_positive = float(matrix[0, 1])
    false_negative = float(matrix[1, 0])
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else 0.0


def balanced_accuracy(matrix: np.ndarray) -> float:
    recalls = []
    for index in range(matrix.shape[0]):
        total = float(matrix[index, :].sum())
        recalls.append(float(matrix[index, index]) / total if total else 0.0)
    return statistics.fmean(recalls)


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    position = 0
    while position < len(scores):
        end = position + 1
        while end < len(scores) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    positive_positions = np.flatnonzero(ordered == 1)
    return float(
        np.mean(cumulative[positive_positions] / (positive_positions + 1))
    )


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    if logits.ndim != 2 or len(labels) != len(logits):
        raise ValueError("Labels and logits have incompatible shapes")
    predictions = logits.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, logits.shape[1])
    metrics = {
        "mcc": matthews_correlation(matrix),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": balanced_accuracy(matrix),
        "macro_f1": macro_f1(matrix),
    }
    if logits.shape[1] == 2 and set(int(value) for value in labels) == {0, 1}:
        metrics["f1"] = positive_class_f1(matrix)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        metrics["roc_auc"] = binary_roc_auc(labels, probabilities[:, 1])
        metrics["average_precision"] = binary_average_precision(
            labels, probabilities[:, 1]
        )
    return metrics
