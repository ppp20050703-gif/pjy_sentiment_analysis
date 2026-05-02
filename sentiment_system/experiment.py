from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .data import (
    LABELS,
    Example,
    find_textbook_data_dir,
    label_distribution,
    load_sentiment_csv,
    load_stopwords,
    stratified_kfold,
    stratified_split,
)
from .features import Vectorizer
from .lexicon import LexiconSentimentAnalyzer
from .metrics import (
    classification_report_text,
    confusion_matrix,
    metrics_row,
    save_confusion_matrix_svg,
)
from .models import LogisticRegressionClassifier, MajorityClassifier, MultinomialNaiveBayes
from .tokenizer import ChineseTokenizer, tokenize_texts


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    seed: int = 42
    test_size: float = 0.2
    cv: int = 5
    max_features: int = 3000
    tokenizer_mode: str = "char_bigram"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    category: str
    vector_weighting: str | None
    model_factory: Callable[[], object]


def default_model_specs(seed: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            key="majority",
            display_name="多数类基线",
            category="实验下限",
            vector_weighting="bow",
            model_factory=MajorityClassifier,
        ),
        ModelSpec(
            key="lexicon_textbook",
            display_name="教材复现：情感词典计分",
            category="教材第9章词典法",
            vector_weighting=None,
            model_factory=lambda: None,
        ),
        ModelSpec(
            key="nb_bow_textbook",
            display_name="教材复现：BoW + 朴素贝叶斯",
            category="教材第9章机器学习情感分类",
            vector_weighting="bow",
            model_factory=MultinomialNaiveBayes,
        ),
        ModelSpec(
            key="nb_tfidf_improved",
            display_name="改进对比：TF-IDF + 朴素贝叶斯",
            category="向量化方法对比",
            vector_weighting="tfidf",
            model_factory=MultinomialNaiveBayes,
        ),
        ModelSpec(
            key="maxent_tfidf_paper",
            display_name="论文方向：TF-IDF + MaxEnt(Logistic)",
            category="Pang等情感分类机器学习路线",
            vector_weighting="tfidf",
            model_factory=lambda: LogisticRegressionClassifier(seed=seed),
        ),
    ]


def run_experiments(config: ExperimentConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_sentiment_csv(config.data_dir / "sentiment.csv")
    train_examples, test_examples = stratified_split(
        examples, test_size=config.test_size, seed=config.seed
    )

    specs = default_model_specs(config.seed)
    rows: list[dict[str, object]] = []
    reports: list[str] = []
    prediction_rows: list[dict[str, str]] = []

    for spec in specs:
        y_true, y_pred = evaluate_spec(spec, train_examples, test_examples, config)
        row = metrics_row(spec.display_name, y_true, y_pred, LABELS)
        row["key"] = spec.key
        row["category"] = spec.category
        cv_scores = cross_validate_spec(spec, examples, config)
        row["cv_accuracy_mean"] = mean([score["accuracy"] for score in cv_scores])
        row["cv_accuracy_std"] = std([score["accuracy"] for score in cv_scores])
        row["cv_macro_f1_mean"] = mean([score["macro_f1"] for score in cv_scores])
        row["cv_macro_f1_std"] = std([score["macro_f1"] for score in cv_scores])
        rows.append(row)

        matrix = confusion_matrix(y_true, y_pred, LABELS)
        save_confusion_matrix_svg(
            matrix,
            LABELS,
            config.output_dir / f"confusion_{spec.key}.svg",
            spec.display_name,
        )
        reports.append(classification_report_text(y_true, y_pred, LABELS, spec.display_name))

        for example, pred in zip(test_examples, y_pred):
            prediction_rows.append(
                {
                    "model": spec.display_name,
                    "text": example.text,
                    "true_label": example.label,
                    "predicted_label": pred,
                }
            )

    rows.sort(key=lambda item: float(item["macro_f1"]), reverse=True)
    write_metrics(config.output_dir / "metrics.csv", rows)
    report_separator = "\n\n" + ("=" * 72) + "\n\n"
    (config.output_dir / "classification_reports.txt").write_text(
        report_separator.join(reports),
        encoding="utf-8",
    )
    write_predictions(config.output_dir / "predictions.csv", prediction_rows)

    summary = {
        "data_path": str(config.data_dir / "sentiment.csv"),
        "dataset_size": len(examples),
        "label_distribution": label_distribution(examples),
        "train_size": len(train_examples),
        "test_size": len(test_examples),
        "tokenizer": config.tokenizer_mode,
        "max_features": config.max_features,
        "best_model": rows[0]["model"],
        "metrics": rows,
    }
    (config.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def evaluate_spec(
    spec: ModelSpec,
    train_examples: list[Example],
    test_examples: list[Example],
    config: ExperimentConfig,
) -> tuple[list[str], list[str]]:
    y_true = [example.label for example in test_examples]
    if spec.key == "lexicon_textbook":
        analyzer = LexiconSentimentAnalyzer.from_data_dir(config.data_dir)
        analyzer.fit([example.text for example in train_examples], [example.label for example in train_examples])
        return y_true, analyzer.predict([example.text for example in test_examples])

    stopwords = load_stopwords(config.data_dir)
    tokenizer = ChineseTokenizer(mode=config.tokenizer_mode, stopwords=stopwords)
    train_tokens = tokenize_texts(tokenizer, [example.text for example in train_examples])
    test_tokens = tokenize_texts(tokenizer, [example.text for example in test_examples])
    vectorizer = Vectorizer(
        weighting=spec.vector_weighting or "bow",
        max_features=config.max_features,
        l2_normalize=True,
    )
    train_vectors = vectorizer.fit_transform(train_tokens)
    test_vectors = vectorizer.transform(test_tokens)
    model = spec.model_factory()
    model.fit(train_vectors, [example.label for example in train_examples])
    return y_true, model.predict(test_vectors)


def cross_validate_spec(
    spec: ModelSpec,
    examples: list[Example],
    config: ExperimentConfig,
) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    folds = stratified_kfold(examples, k=config.cv, seed=config.seed)
    for train_examples, valid_examples in folds:
        y_true, y_pred = evaluate_spec(spec, train_examples, valid_examples, config)
        row = metrics_row(spec.key, y_true, y_pred, LABELS)
        scores.append(
            {
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
            }
        )
    return scores


def train_for_prediction(
    model_key: str,
    examples: list[Example],
    config: ExperimentConfig,
) -> Callable[[list[str]], list[str]]:
    specs = {spec.key: spec for spec in default_model_specs(config.seed)}
    if model_key not in specs:
        raise ValueError(f"未知模型: {model_key}; 可选: {', '.join(specs)}")
    spec = specs[model_key]
    labels = [example.label for example in examples]
    if spec.key == "lexicon_textbook":
        analyzer = LexiconSentimentAnalyzer.from_data_dir(config.data_dir)
        analyzer.fit([example.text for example in examples], labels)
        return analyzer.predict

    stopwords = load_stopwords(config.data_dir)
    tokenizer = ChineseTokenizer(mode=config.tokenizer_mode, stopwords=stopwords)
    train_tokens = tokenize_texts(tokenizer, [example.text for example in examples])
    vectorizer = Vectorizer(
        weighting=spec.vector_weighting or "bow",
        max_features=config.max_features,
        l2_normalize=True,
    )
    vectors = vectorizer.fit_transform(train_tokens)
    model = spec.model_factory()
    model.fit(vectors, labels)

    def predict(texts: list[str]) -> list[str]:
        tokens = tokenize_texts(tokenizer, texts)
        test_vectors = vectorizer.transform(tokens)
        return model.predict(test_vectors)

    return predict


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
            writer.writerow(
                {
                    name: format_metric(row.get(name, ""))
                    for name in fieldnames
                }
            )


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "text", "true_label", "predicted_label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def make_config(
    data_dir: str | None,
    output_dir: str | None,
    seed: int,
    test_size: float,
    cv: int,
    max_features: int,
    tokenizer_mode: str,
) -> ExperimentConfig:
    resolved_data_dir = Path(data_dir) if data_dir else find_textbook_data_dir()
    resolved_output_dir = Path(output_dir) if output_dir else Path("outputs")
    return ExperimentConfig(
        data_dir=resolved_data_dir,
        output_dir=resolved_output_dir,
        seed=seed,
        test_size=test_size,
        cv=cv,
        max_features=max_features,
        tokenizer_mode=tokenizer_mode,
    )
