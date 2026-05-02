from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .data import NEGATIVE, POSITIVE, load_boson_lexicon, load_degree_words, load_negations


@dataclass(frozen=True)
class LexiconHit:
    start: int
    word: str
    kind: str
    value: float


class LexiconSentimentAnalyzer:
    """Dictionary sentiment scorer based on the textbook's lexicon method."""

    def __init__(
        self,
        sentiment_lexicon: dict[str, float],
        negations: set[str],
        degree_words: dict[str, float],
        max_match_length: int = 8,
        context_window: int = 6,
    ) -> None:
        self.sentiment_lexicon = {
            word: score
            for word, score in sentiment_lexicon.items()
            if 0 < len(word) <= max_match_length
        }
        self.negations = {word for word in negations if 0 < len(word) <= max_match_length}
        self.degree_words = {
            word: value
            for word, value in degree_words.items()
            if 0 < len(word) <= max_match_length
        }
        self.max_match_length = max_match_length
        self.context_window = context_window
        self.tie_label = POSITIVE

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "LexiconSentimentAnalyzer":
        return cls(
            sentiment_lexicon=load_boson_lexicon(data_dir),
            negations=load_negations(data_dir),
            degree_words=load_degree_words(data_dir),
        )

    def fit(self, texts: list[str], labels: list[str]) -> "LexiconSentimentAnalyzer":
        del texts
        self.tie_label = Counter(labels).most_common(1)[0][0]
        return self

    def scan(self, text: str) -> list[LexiconHit]:
        hits: list[LexiconHit] = []
        i = 0
        while i < len(text):
            matched: LexiconHit | None = None
            max_len = min(self.max_match_length, len(text) - i)
            for size in range(max_len, 0, -1):
                word = text[i : i + size]
                if word in self.sentiment_lexicon:
                    matched = LexiconHit(i, word, "sentiment", self.sentiment_lexicon[word])
                    break
                if word in self.negations:
                    matched = LexiconHit(i, word, "negation", -1.0)
                    break
                if word in self.degree_words:
                    matched = LexiconHit(i, word, "degree", self.degree_words[word])
                    break
            if matched is None:
                i += 1
            else:
                hits.append(matched)
                i += max(1, len(matched.word))
        return hits

    def score(self, text: str) -> float:
        hits = self.scan(text)
        total = 0.0
        for idx, hit in enumerate(hits):
            if hit.kind != "sentiment":
                continue
            factor = 1.0
            for previous in hits[max(0, idx - 4) : idx]:
                if hit.start - previous.start > self.context_window:
                    continue
                if previous.kind == "negation":
                    factor *= -1.0
                elif previous.kind == "degree":
                    factor *= previous.value
            total += factor * hit.value
        return total

    def predict_one(self, text: str) -> str:
        score = self.score(text)
        if score > 0:
            return POSITIVE
        if score < 0:
            return NEGATIVE
        return self.tie_label

    def predict(self, texts: list[str]) -> list[str]:
        return [self.predict_one(text) for text in texts]

    def explain(self, text: str, limit: int = 8) -> list[LexiconHit]:
        return self.scan(text)[:limit]
