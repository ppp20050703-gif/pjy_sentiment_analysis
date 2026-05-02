from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Iterable

from .data import LABELS, NEGATIVE, POSITIVE
from .features import SparseVector


class MajorityClassifier:
    def __init__(self) -> None:
        self.majority_label = POSITIVE

    def fit(self, vectors: list[SparseVector], labels: list[str]) -> "MajorityClassifier":
        del vectors
        self.majority_label = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, vectors: list[SparseVector]) -> list[str]:
        return [self.majority_label for _ in vectors]


class MultinomialNaiveBayes:
    """Multinomial NB, corresponding to the textbook's NB sentiment classifier."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_log_prior_: dict[str, float] = {}
        self.feature_log_prob_: dict[str, dict[int, float]] = {}
        self.default_log_prob_: dict[str, float] = {}
        self.n_features_ = 0

    def fit(self, vectors: list[SparseVector], labels: list[str]) -> "MultinomialNaiveBayes":
        self.n_features_ = 1 + max((idx for vector in vectors for idx in vector), default=-1)
        class_counts = Counter(labels)
        total_docs = len(labels)
        feature_counts: dict[str, defaultdict[int, float]] = {
            label: defaultdict(float) for label in class_counts
        }
        feature_totals: defaultdict[str, float] = defaultdict(float)

        for vector, label in zip(vectors, labels):
            for idx, value in vector.items():
                feature_counts[label][idx] += value
                feature_totals[label] += value

        self.class_log_prior_ = {
            label: math.log(class_counts[label] / total_docs)
            for label in class_counts
        }
        self.feature_log_prob_ = {}
        self.default_log_prob_ = {}
        for label in class_counts:
            denominator = feature_totals[label] + self.alpha * max(1, self.n_features_)
            self.default_log_prob_[label] = math.log(self.alpha / denominator)
            self.feature_log_prob_[label] = {
                idx: math.log((value + self.alpha) / denominator)
                for idx, value in feature_counts[label].items()
            }
        return self

    def predict_one(self, vector: SparseVector) -> str:
        scores: dict[str, float] = {}
        for label, prior in self.class_log_prior_.items():
            score = prior
            default = self.default_log_prob_[label]
            probs = self.feature_log_prob_[label]
            for idx, value in vector.items():
                score += value * probs.get(idx, default)
            scores[label] = score
        return max(scores.items(), key=lambda item: item[1])[0]

    def predict(self, vectors: list[SparseVector]) -> list[str]:
        return [self.predict_one(vector) for vector in vectors]


class LogisticRegressionClassifier:
    """Binary MaxEnt/logistic classifier trained by SGD.

    This gives the paper-direction comparison a lightweight implementation:
    sentiment classification with sparse text features and a maximum-entropy
    style linear decision function.
    """

    def __init__(
        self,
        epochs: int = 35,
        learning_rate: float = 0.35,
        l2: float = 0.0005,
        seed: int = 42,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.seed = seed
        self.weights_: defaultdict[int, float] = defaultdict(float)
        self.bias_: float = 0.0

    def fit(self, vectors: list[SparseVector], labels: list[str]) -> "LogisticRegressionClassifier":
        rng = random.Random(self.seed)
        indices = list(range(len(vectors)))
        for epoch in range(self.epochs):
            rng.shuffle(indices)
            lr = self.learning_rate / (1.0 + 0.05 * epoch)
            for idx in indices:
                vector = vectors[idx]
                target = 1.0 if labels[idx] == POSITIVE else 0.0
                probability = self.predict_proba_one(vector)
                error = probability - target
                self.bias_ -= lr * error
                for feature_idx, value in vector.items():
                    self.weights_[feature_idx] -= lr * (
                        error * value + self.l2 * self.weights_[feature_idx]
                    )
        return self

    def decision_function(self, vector: SparseVector) -> float:
        return self.bias_ + sum(self.weights_[idx] * value for idx, value in vector.items())

    def predict_proba_one(self, vector: SparseVector) -> float:
        z = self.decision_function(vector)
        if z >= 0:
            exp_neg = math.exp(-z)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(z)
        return exp_pos / (1.0 + exp_pos)

    def predict(self, vectors: list[SparseVector]) -> list[str]:
        return [
            POSITIVE if self.predict_proba_one(vector) >= 0.5 else NEGATIVE
            for vector in vectors
        ]


def labels_from_examples(examples: Iterable[object]) -> list[str]:
    return [getattr(example, "label") for example in examples]


def stable_labels(labels: Iterable[str]) -> list[str]:
    labels_set = set(labels)
    return [label for label in LABELS if label in labels_set]
