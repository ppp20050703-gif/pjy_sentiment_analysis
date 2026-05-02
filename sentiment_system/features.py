from __future__ import annotations

import math
from collections import Counter


SparseVector = dict[int, float]


class Vectorizer:
    """Bag-of-words / TF-IDF vectorizer implemented with the standard library."""

    def __init__(
        self,
        weighting: str = "tfidf",
        min_df: int = 1,
        max_features: int = 3000,
        l2_normalize: bool = True,
    ) -> None:
        if weighting not in {"bow", "tfidf"}:
            raise ValueError("weighting 必须是 bow 或 tfidf")
        self.weighting = weighting
        self.min_df = min_df
        self.max_features = max_features
        self.l2_normalize = l2_normalize
        self.vocabulary_: dict[str, int] = {}
        self.idf_: dict[int, float] = {}

    def fit(self, tokenized_docs: list[list[str]]) -> "Vectorizer":
        doc_freq: Counter[str] = Counter()
        total_freq: Counter[str] = Counter()
        for tokens in tokenized_docs:
            counts = Counter(tokens)
            doc_freq.update(counts.keys())
            total_freq.update(counts)

        terms = [
            term
            for term, df in doc_freq.items()
            if df >= self.min_df and term.strip()
        ]
        terms.sort(key=lambda term: (-doc_freq[term], -total_freq[term], term))
        if self.max_features:
            terms = terms[: self.max_features]

        self.vocabulary_ = {term: idx for idx, term in enumerate(terms)}
        n_docs = max(1, len(tokenized_docs))
        self.idf_ = {
            idx: math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
            for term, idx in self.vocabulary_.items()
        }
        return self

    def transform(self, tokenized_docs: list[list[str]]) -> list[SparseVector]:
        vectors: list[SparseVector] = []
        for tokens in tokenized_docs:
            counts: Counter[int] = Counter()
            for token in tokens:
                idx = self.vocabulary_.get(token)
                if idx is not None:
                    counts[idx] += 1

            if not counts:
                vectors.append({})
                continue

            if self.weighting == "bow":
                vector = {idx: float(value) for idx, value in counts.items()}
            else:
                token_total = sum(counts.values())
                vector = {
                    idx: (value / token_total) * self.idf_.get(idx, 1.0)
                    for idx, value in counts.items()
                }

            if self.l2_normalize:
                norm = math.sqrt(sum(value * value for value in vector.values()))
                if norm > 0:
                    vector = {idx: value / norm for idx, value in vector.items()}
            vectors.append(vector)
        return vectors

    def fit_transform(self, tokenized_docs: list[list[str]]) -> list[SparseVector]:
        return self.fit(tokenized_docs).transform(tokenized_docs)

    @property
    def n_features(self) -> int:
        return len(self.vocabulary_)
