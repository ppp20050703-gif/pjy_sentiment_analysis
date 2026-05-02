from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from .emotion import EMOTION_LABELS, emotion_distribution, load_multiclass_emotion_dataset
from .experiment import ExperimentConfig, format_metric
from .metrics import confusion_matrix, metrics_row, save_confusion_matrix_svg
from .tokenizer import ChineseTokenizer


class TokenizerAnalyzer:
    """Pickle-friendly analyzer used by sklearn vectorizers."""

    def __init__(self, tokenizer_mode: str) -> None:
        self.tokenizer_mode = tokenizer_mode
        self._tokenizer: ChineseTokenizer | None = None

    def __call__(self, text: str) -> list[str]:
        if self._tokenizer is None:
            self._tokenizer = ChineseTokenizer(mode=self.tokenizer_mode)
        return self._tokenizer.tokenize(text)

    def __getstate__(self) -> dict[str, str]:
        return {"tokenizer_mode": self.tokenizer_mode}

    def __setstate__(self, state: dict[str, str]) -> None:
        self.tokenizer_mode = state["tokenizer_mode"]
        self._tokenizer = None


def add_vendor_path() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    vendor = project_dir / "vendor"
    if vendor.exists():
        sys.path.insert(0, str(vendor))


def sklearn_available() -> bool:
    add_vendor_path()
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def run_sklearn_multiclass_experiments(config: ExperimentConfig) -> dict[str, Any]:
    output_dir = config.output_dir / "sklearn_multiclass"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sklearn_available():
        status = {
            "enabled": False,
            "reason": "当前 Python 环境未安装 scikit-learn。运行 python -m pip install --target vendor scikit-learn jieba pandas matplotlib joblib 后可启用。",
        }
        (output_dir / "sklearn_dependency_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return status

    from joblib import dump
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_validate, train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    examples = load_multiclass_emotion_dataset(config.data_dir)
    texts = [example.text for example in examples]
    labels = [example.label for example in examples]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels,
    )

    analyzer = TokenizerAnalyzer(config.tokenizer_mode)

    models = [
        (
            "sk_nb_tfidf",
            "sklearn多分类：TF-IDF + MultinomialNB",
            MultinomialNB(alpha=0.5),
        ),
        (
            "sk_linear_svc",
            "sklearn多分类：TF-IDF + LinearSVC",
            LinearSVC(class_weight="balanced", random_state=config.seed),
        ),
        (
            "sk_logistic",
            "sklearn多分类：TF-IDF + LogisticRegression",
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                solver="lbfgs",
                random_state=config.seed,
            ),
        ),
        (
            "sk_sgd",
            "sklearn多分类：TF-IDF + SGDClassifier",
            SGDClassifier(
                loss="log_loss",
                alpha=0.0003,
                max_iter=1500,
                class_weight="balanced",
                random_state=config.seed,
            ),
        ),
        (
            "sk_random_forest",
            "sklearn多分类：TF-IDF + RandomForest",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=config.seed,
                n_jobs=-1,
            ),
        ),
    ]

    rows: list[dict[str, object]] = []
    reports: list[str] = []
    best_pipeline: Pipeline | None = None
    best_macro_f1 = -1.0
    best_key = ""

    for key, display_name, classifier in models:
        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer=analyzer,
                        max_features=config.max_features,
                        sublinear_tf=True,
                        norm="l2",
                    ),
                ),
                ("clf", classifier),
            ]
        )
        pipeline.fit(train_texts, train_labels)
        predictions = list(pipeline.predict(test_texts))
        row = metrics_row(display_name, test_labels, predictions, EMOTION_LABELS)
        row["key"] = key
        row["category"] = "scikit-learn机器学习库模型"

        cv_result = cross_validate(
            pipeline,
            texts,
            labels,
            cv=config.cv,
            scoring=["accuracy", "f1_macro"],
            n_jobs=None,
        )
        row["cv_accuracy_mean"] = float(cv_result["test_accuracy"].mean())
        row["cv_accuracy_std"] = float(cv_result["test_accuracy"].std())
        row["cv_macro_f1_mean"] = float(cv_result["test_f1_macro"].mean())
        row["cv_macro_f1_std"] = float(cv_result["test_f1_macro"].std())
        rows.append(row)

        matrix = confusion_matrix(test_labels, predictions, EMOTION_LABELS)
        save_confusion_matrix_svg(
            matrix,
            EMOTION_LABELS,
            output_dir / f"confusion_{key}.svg",
            display_name,
        )
        reports.append(
            display_name
            + "\n\n"
            + classification_report(
                test_labels,
                predictions,
                labels=EMOTION_LABELS,
                zero_division=0,
            )
        )
        if float(row["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(row["macro_f1"])
            best_pipeline = pipeline
            best_key = key

    rows.sort(key=lambda item: float(item["macro_f1"]), reverse=True)
    write_metrics(output_dir / "sklearn_multiclass_metrics.csv", rows)
    report_separator = "\n\n" + ("=" * 72) + "\n\n"
    (output_dir / "sklearn_classification_reports.txt").write_text(
        report_separator.join(reports),
        encoding="utf-8",
    )
    if best_pipeline is not None:
        dump(best_pipeline, output_dir / "best_sklearn_emotion_model.joblib")

    summary = {
        "enabled": True,
        "task": "scikit-learn多类别细粒度情绪分类",
        "labels": EMOTION_LABELS,
        "dataset_size": len(examples),
        "label_distribution": emotion_distribution(examples),
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "best_model_key": best_key,
        "best_macro_f1": best_macro_f1,
        "metrics": rows,
    }
    (output_dir / "sklearn_multiclass_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


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
