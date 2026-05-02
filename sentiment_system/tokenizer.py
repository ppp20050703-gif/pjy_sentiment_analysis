from __future__ import annotations

import re
from typing import Iterable


_CHINESE_OR_ALNUM = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+")


class ChineseTokenizer:
    """Small tokenizer for environments without jieba.

    The coursework machine may not have third-party NLP packages installed.
    This tokenizer uses Chinese character unigrams and adjacent bigrams, which
    keeps important polarity clues such as "不好", "糟糕", "很棒" available to
    the models. If jieba is installed, mode="jieba" can be used instead.
    """

    def __init__(
        self,
        mode: str = "char_bigram",
        stopwords: Iterable[str] | None = None,
    ) -> None:
        self.mode = mode
        self.stopwords = set(stopwords or [])
        self._jieba = None
        if mode == "jieba":
            try:
                import jieba  # type: ignore

                self._jieba = jieba
            except Exception:
                self.mode = "char_bigram"

    def tokenize(self, text: str) -> list[str]:
        if self.mode == "jieba" and self._jieba is not None:
            tokens = [token.strip() for token in self._jieba.lcut(text)]
            return [token for token in tokens if token and token not in self.stopwords]

        units = _CHINESE_OR_ALNUM.findall(text)
        units = [unit for unit in units if unit and unit not in self.stopwords]
        if self.mode == "char":
            return units

        tokens = units[:]
        for left, right in zip(units, units[1:]):
            if self._is_chinese_char(left) and self._is_chinese_char(right):
                bigram = left + right
                if bigram not in self.stopwords:
                    tokens.append(bigram)
        return tokens

    @staticmethod
    def _is_chinese_char(token: str) -> bool:
        return len(token) == 1 and "\u4e00" <= token <= "\u9fff"


def tokenize_texts(tokenizer: ChineseTokenizer, texts: Iterable[str]) -> list[list[str]]:
    return [tokenizer.tokenize(text) for text in texts]
