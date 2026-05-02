from __future__ import annotations

import html
from pathlib import Path


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> list[list[int]]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            matrix[label_to_idx[true]][label_to_idx[pred]] += 1
    return matrix


def precision_recall_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    matrix = confusion_matrix(y_true, y_pred, labels)
    result: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(len(labels)) if row != i)
        fn = sum(matrix[i][col] for col in range(len(labels)) if col != i)
        support = sum(matrix[i])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }

    result["macro_avg"] = {
        "precision": sum(result[label]["precision"] for label in labels) / len(labels),
        "recall": sum(result[label]["recall"] for label in labels) / len(labels),
        "f1": sum(result[label]["f1"] for label in labels) / len(labels),
        "support": float(len(y_true)),
    }
    result["accuracy"] = {
        "precision": accuracy_score(y_true, y_pred),
        "recall": accuracy_score(y_true, y_pred),
        "f1": accuracy_score(y_true, y_pred),
        "support": float(len(y_true)),
    }
    return result


def metrics_row(model_name: str, y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, float | str]:
    report = precision_recall_f1(y_true, y_pred, labels)
    return {
        "model": model_name,
        "accuracy": report["accuracy"]["f1"],
        "macro_precision": report["macro_avg"]["precision"],
        "macro_recall": report["macro_avg"]["recall"],
        "macro_f1": report["macro_avg"]["f1"],
    }


def classification_report_text(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    title: str,
) -> str:
    report = precision_recall_f1(y_true, y_pred, labels)
    lines = [title, ""]
    lines.append(f"{'label':<12}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>12}")
    for label in labels:
        item = report[label]
        lines.append(
            f"{label:<12}{item['precision']:>12.4f}{item['recall']:>12.4f}"
            f"{item['f1']:>12.4f}{int(item['support']):>12}"
        )
    item = report["macro_avg"]
    lines.append(
        f"{'macro_avg':<12}{item['precision']:>12.4f}{item['recall']:>12.4f}"
        f"{item['f1']:>12.4f}{int(item['support']):>12}"
    )
    lines.append(f"accuracy: {report['accuracy']['f1']:.4f}")
    return "\n".join(lines)


def save_confusion_matrix_svg(
    matrix: list[list[int]],
    labels: list[str],
    path: Path,
    title: str,
) -> None:
    cell = 90
    margin_left = 95
    margin_top = 70
    width = margin_left + cell * len(labels) + 35
    height = margin_top + cell * len(labels) + 60
    max_value = max((value for row in matrix for value in row), default=1) or 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="18" font-family="Arial, SimHei">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="13" font-family="Arial, SimHei">预测标签</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-size="13" font-family="Arial, SimHei">真实标签</text>',
    ]

    for j, label in enumerate(labels):
        x = margin_left + j * cell + cell / 2
        parts.append(
            f'<text x="{x}" y="{margin_top - 18}" text-anchor="middle" font-size="14" font-family="Arial, SimHei">{html.escape(label)}</text>'
        )
    for i, label in enumerate(labels):
        y = margin_top + i * cell + cell / 2 + 5
        parts.append(
            f'<text x="{margin_left - 18}" y="{y}" text-anchor="end" font-size="14" font-family="Arial, SimHei">{html.escape(label)}</text>'
        )

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            x = margin_left + j * cell
            y = margin_top + i * cell
            intensity = int(245 - 150 * (value / max_value))
            fill = f"rgb({intensity},{intensity + 5},{255})"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#334155"/>')
            parts.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 6}" text-anchor="middle" font-size="22" font-family="Arial, SimHei" fill="#0f172a">{value}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
