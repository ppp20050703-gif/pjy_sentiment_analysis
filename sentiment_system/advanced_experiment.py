from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .data import load_stopwords, stratified_kfold, stratified_split
from .emotion import (
    DISAPPOINTMENT,
    EMOTION_LABELS,
    NEUTRAL,
    EmotionExample,
    build_multiclass_emotion_dataset,
    emotion_distribution,
    explain_labeling_rules,
    load_multiclass_emotion_dataset,
)
from .experiment import ExperimentConfig, format_metric
from .features import Vectorizer
from .metrics import (
    classification_report_text,
    confusion_matrix,
    metrics_row,
    save_confusion_matrix_svg,
)
from .models import MajorityClassifier, MultinomialNaiveBayes
from .tokenizer import ChineseTokenizer, tokenize_texts


@dataclass(frozen=True)
class AdvancedModelSpec:
    key: str
    display_name: str
    category: str
    vector_weighting: str | None
    model_factory: Callable[[], object]


class KeywordEmotionClassifier:
    """Interpretable rule classifier for fine-grained emotion labels."""

    def fit(self, texts: list[str], labels: list[str]) -> "KeywordEmotionClassifier":
        del texts
        self.default_label = max(set(labels), key=labels.count)
        return self

    def predict_one(self, text: str) -> str:
        if any(token in text for token in ("没有明显情绪", "客观", "只是提到", "正常状态", "继续观察")):
            return NEUTRAL
        rules = [
            ("喜悦", ("愉快", "开心", "精彩", "好吃", "好玩", "很美", "很棒", "棒", "美")),
            ("满意", ("贴心", "方便", "好用", "很好", "不错", "出色", "很长", "质量好")),
            ("赞赏", ("推荐", "赞", "优秀", "完美", "值得", "喜欢")),
            ("愤怒", ("差劲", "糟糕", "太差", "极差", "差", "糟")),
            ("失望", ("失望", "不满意", "不好", "一般", "短", "不值")),
            ("厌恶", ("无聊", "讨厌", "恶心", "垃圾", "厌恶")),
        ]
        for label, keywords in rules:
            if any(keyword in text for keyword in keywords):
                return label
        return getattr(self, "default_label", DISAPPOINTMENT)

    def predict(self, texts: list[str]) -> list[str]:
        return [self.predict_one(text) for text in texts]


def advanced_model_specs() -> list[AdvancedModelSpec]:
    return [
        AdvancedModelSpec(
            key="emotion_majority",
            display_name="多类别基线：多数类",
            category="实验下限",
            vector_weighting="bow",
            model_factory=MajorityClassifier,
        ),
        AdvancedModelSpec(
            key="emotion_rules",
            display_name="多类别规则模型：情绪关键词",
            category="可解释规则基线",
            vector_weighting=None,
            model_factory=KeywordEmotionClassifier,
        ),
        AdvancedModelSpec(
            key="emotion_nb_bow",
            display_name="多类别机器学习：BoW + 朴素贝叶斯",
            category="教材机器学习方法扩展",
            vector_weighting="bow",
            model_factory=MultinomialNaiveBayes,
        ),
        AdvancedModelSpec(
            key="emotion_nb_tfidf",
            display_name="多类别机器学习：TF-IDF + 朴素贝叶斯",
            category="向量化方法扩展对比",
            vector_weighting="tfidf",
            model_factory=MultinomialNaiveBayes,
        ),
    ]


def run_multiclass_experiments(config: ExperimentConfig) -> dict[str, object]:
    output_dir = config.output_dir / "advanced_multiclass"
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_multiclass_emotion_dataset(config.data_dir)
    write_multiclass_dataset(output_dir / "multiclass_emotion_dataset.csv", examples)
    write_labeling_rules(output_dir / "emotion_labeling_rules.csv")

    train_examples, test_examples = stratified_split(
        examples, test_size=config.test_size, seed=config.seed
    )
    rows: list[dict[str, object]] = []
    reports: list[str] = []
    prediction_rows: list[dict[str, str]] = []

    for spec in advanced_model_specs():
        y_true, y_pred = evaluate_advanced_spec(spec, train_examples, test_examples, config)
        row = metrics_row(spec.display_name, y_true, y_pred, EMOTION_LABELS)
        row["key"] = spec.key
        row["category"] = spec.category
        cv_scores = cross_validate_advanced_spec(spec, examples, config)
        row["cv_accuracy_mean"] = mean([score["accuracy"] for score in cv_scores])
        row["cv_accuracy_std"] = std([score["accuracy"] for score in cv_scores])
        row["cv_macro_f1_mean"] = mean([score["macro_f1"] for score in cv_scores])
        row["cv_macro_f1_std"] = std([score["macro_f1"] for score in cv_scores])
        rows.append(row)

        matrix = confusion_matrix(y_true, y_pred, EMOTION_LABELS)
        save_confusion_matrix_svg(
            matrix,
            EMOTION_LABELS,
            output_dir / f"confusion_{spec.key}.svg",
            spec.display_name,
        )
        reports.append(classification_report_text(y_true, y_pred, EMOTION_LABELS, spec.display_name))

        for example, pred in zip(test_examples, y_pred):
            prediction_rows.append(
                {
                    "model": spec.display_name,
                    "text": example.text,
                    "true_label": example.label,
                    "predicted_label": pred,
                    "source": example.source,
                }
            )

    rows.sort(key=lambda item: float(item["macro_f1"]), reverse=True)
    write_metrics(output_dir / "multiclass_metrics.csv", rows)
    write_predictions(output_dir / "multiclass_predictions.csv", prediction_rows)
    report_separator = "\n\n" + ("=" * 72) + "\n\n"
    (output_dir / "multiclass_classification_reports.txt").write_text(
        report_separator.join(reports),
        encoding="utf-8",
    )

    summary = {
        "task": "多类别细粒度情绪分类",
        "labels": EMOTION_LABELS,
        "dataset_size": len(examples),
        "label_distribution": emotion_distribution(examples),
        "train_size": len(train_examples),
        "test_size": len(test_examples),
        "labeling_method": "教材二分类数据 + 情绪关键词弱标注 + 中性模板补充",
        "best_model": rows[0]["model"],
        "metrics": rows,
    }
    (output_dir / "multiclass_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def evaluate_advanced_spec(
    spec: AdvancedModelSpec,
    train_examples: list[EmotionExample],
    test_examples: list[EmotionExample],
    config: ExperimentConfig,
) -> tuple[list[str], list[str]]:
    y_true = [example.label for example in test_examples]
    train_texts = [example.text for example in train_examples]
    test_texts = [example.text for example in test_examples]

    if spec.vector_weighting is None:
        model = spec.model_factory()
        model.fit(train_texts, [example.label for example in train_examples])
        return y_true, model.predict(test_texts)

    stopwords = load_stopwords(config.data_dir)
    tokenizer = ChineseTokenizer(mode=config.tokenizer_mode, stopwords=stopwords)
    train_tokens = tokenize_texts(tokenizer, train_texts)
    test_tokens = tokenize_texts(tokenizer, test_texts)
    vectorizer = Vectorizer(
        weighting=spec.vector_weighting,
        max_features=config.max_features,
        l2_normalize=True,
    )
    train_vectors = vectorizer.fit_transform(train_tokens)
    test_vectors = vectorizer.transform(test_tokens)
    model = spec.model_factory()
    model.fit(train_vectors, [example.label for example in train_examples])
    return y_true, model.predict(test_vectors)


def cross_validate_advanced_spec(
    spec: AdvancedModelSpec,
    examples: list[EmotionExample],
    config: ExperimentConfig,
) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    for train_examples, valid_examples in stratified_kfold(examples, k=config.cv, seed=config.seed):
        y_true, y_pred = evaluate_advanced_spec(spec, train_examples, valid_examples, config)
        row = metrics_row(spec.key, y_true, y_pred, EMOTION_LABELS)
        scores.append(
            {
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
            }
        )
    return scores


def train_multiclass_predictor(
    model_key: str,
    config: ExperimentConfig,
) -> Callable[[list[str]], list[str]]:
    specs = {spec.key: spec for spec in advanced_model_specs()}
    if model_key not in specs:
        raise ValueError(f"未知模型: {model_key}; 可选: {', '.join(specs)}")
    examples = load_multiclass_emotion_dataset(config.data_dir)
    spec = specs[model_key]
    train_texts = [example.text for example in examples]
    labels = [example.label for example in examples]

    if spec.vector_weighting is None:
        model = spec.model_factory()
        model.fit(train_texts, labels)
        return model.predict

    stopwords = load_stopwords(config.data_dir)
    tokenizer = ChineseTokenizer(mode=config.tokenizer_mode, stopwords=stopwords)
    train_tokens = tokenize_texts(tokenizer, train_texts)
    vectorizer = Vectorizer(
        weighting=spec.vector_weighting,
        max_features=config.max_features,
        l2_normalize=True,
    )
    vectors = vectorizer.fit_transform(train_tokens)
    model = spec.model_factory()
    model.fit(vectors, labels)

    def predict(texts: list[str]) -> list[str]:
        tokens = tokenize_texts(tokenizer, texts)
        return model.predict(vectorizer.transform(tokens))

    return predict


def write_multiclass_dataset(path: Path, examples: list[EmotionExample]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "emotion_label", "polarity", "source"])
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "text": example.text,
                    "emotion_label": example.label,
                    "polarity": example.polarity,
                    "source": example.source,
                }
            )


def write_labeling_rules(path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "keywords"])
        writer.writeheader()
        writer.writerows(explain_labeling_rules())


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "key",
        "model",
        "category",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: format_metric(row.get(name, "")) for name in fieldnames})


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "text", "true_label", "predicted_label", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))
