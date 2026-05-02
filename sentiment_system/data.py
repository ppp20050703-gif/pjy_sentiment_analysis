from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


POSITIVE = "正面"
NEGATIVE = "负面"
LABELS = [NEGATIVE, POSITIVE]


@dataclass(frozen=True)
class Example:
    text: str
    label: str


def find_textbook_data_dir(start: Path | None = None) -> Path:
    """Find the textbook data directory from this project or the cwd."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start)
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve())

    for base in candidates:
        for parent in [base, *base.parents]:
            data_dir = parent / "文本情感分析教材代码" / "data"
            if (data_dir / "sentiment.csv").exists():
                return data_dir
    raise FileNotFoundError("未找到 文本情感分析教材代码/data/sentiment.csv")


def normalize_label(label: str) -> str:
    label = label.strip().replace("\ufeff", "")
    if label in {"正面", "positive", "POSITIVE", "1", "pos"}:
        return POSITIVE
    if label in {"负面", "negative", "NEGATIVE", "0", "neg"}:
        return NEGATIVE
    raise ValueError(f"未知情感标签: {label!r}")


def load_sentiment_csv(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("Text") or row.get("text") or "").strip()
            label = normalize_label(row.get("Sentiment") or row.get("label") or "")
            if text:
                examples.append(Example(text=text, label=label))
    if not examples:
        raise ValueError(f"数据文件为空: {path}")
    return examples


def load_word_list(path: Path) -> list[str]:
    words: list[str] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            word = line.strip()
            if word:
                words.append(word)
    return words


def load_stopwords(data_dir: Path) -> set[str]:
    path = data_dir / "stopwords.txt"
    return set(load_word_list(path)) if path.exists() else set()


def load_negations(data_dir: Path) -> set[str]:
    path = data_dir / "否定词.txt"
    return set(load_word_list(path)) if path.exists() else {"不", "没", "无", "非"}


def load_degree_words(data_dir: Path) -> dict[str, float]:
    path = data_dir / "程度副词（中文）.txt"
    degree: dict[str, float] = {}
    if not path.exists():
        return {"很": 1.5, "非常": 2.0, "太": 1.8, "极": 2.0}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    degree[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    return degree


def load_boson_lexicon(data_dir: Path, min_abs_score: float = 0.05) -> dict[str, float]:
    path = data_dir / "BosonNLP_sentiment_score.txt"
    lexicon: dict[str, float] = {}
    if not path.exists():
        return lexicon
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word, value = parts[0], parts[-1]
            try:
                score = float(value)
            except ValueError:
                continue
            if word and abs(score) >= min_abs_score:
                lexicon[word] = score
    return lexicon


def stratified_split(
    examples: list[Example],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        groups[example.label].append(example)

    train: list[Example] = []
    test: list[Example] = []
    for label_examples in groups.values():
        shuffled = label_examples[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_size)))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def stratified_kfold(
    examples: list[Example],
    k: int = 5,
    seed: int = 42,
) -> list[tuple[list[Example], list[Example]]]:
    if k < 2:
        raise ValueError("k 必须大于等于 2")
    rng = random.Random(seed)
    by_label: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    folds: list[list[Example]] = [[] for _ in range(k)]
    for label_examples in by_label.values():
        shuffled = label_examples[:]
        rng.shuffle(shuffled)
        for idx, example in enumerate(shuffled):
            folds[idx % k].append(example)

    result: list[tuple[list[Example], list[Example]]] = []
    for i in range(k):
        valid = folds[i][:]
        train = [example for j, fold in enumerate(folds) if j != i for example in fold]
        rng.shuffle(train)
        rng.shuffle(valid)
        result.append((train, valid))
    return result


def label_distribution(examples: Iterable[Example]) -> dict[str, int]:
    return dict(Counter(example.label for example in examples))
