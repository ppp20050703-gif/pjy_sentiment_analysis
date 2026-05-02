from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .data import Example, NEGATIVE, POSITIVE, load_sentiment_csv


JOY = "喜悦"
SATISFACTION = "满意"
PRAISE = "赞赏"
DISAPPOINTMENT = "失望"
ANGER = "愤怒"
DISGUST = "厌恶"
NEUTRAL = "中性"

EMOTION_LABELS = [JOY, SATISFACTION, PRAISE, DISAPPOINTMENT, ANGER, DISGUST, NEUTRAL]


@dataclass(frozen=True)
class EmotionExample:
    text: str
    label: str
    polarity: str
    source: str


POSITIVE_RULES: list[tuple[str, tuple[str, ...]]] = [
    (JOY, ("愉快", "开心", "精彩", "好吃", "好玩", "很美", "很棒", "棒", "美")),
    (SATISFACTION, ("贴心", "方便", "好用", "很好", "不错", "出色", "很长", "质量好")),
    (PRAISE, ("推荐", "赞", "优秀", "完美", "值得", "喜欢")),
]

NEGATIVE_RULES: list[tuple[str, tuple[str, ...]]] = [
    (ANGER, ("差劲", "糟糕", "太差", "极差", "差", "糟")),
    (DISAPPOINTMENT, ("失望", "不满意", "不好", "一般", "短", "不值")),
    (DISGUST, ("无聊", "讨厌", "恶心", "垃圾", "厌恶")),
]


NEUTRAL_TOPICS = ["电影", "餐厅", "酒店", "手机", "软件", "咖啡馆", "书", "旅行", "项目", "服务"]
NEUTRAL_PATTERNS = [
    "今天记录了{topic}的使用情况",
    "我查看了这个{topic}的基本信息",
    "这次{topic}体验没有明显情绪变化",
    "关于{topic}的描述比较客观",
    "用户只是提到了{topic}，没有表达态度",
    "这个{topic}目前保持正常状态",
    "我了解了一下{topic}的相关内容",
    "{topic}的情况需要继续观察",
]

EMOTION_AUGMENT_TEMPLATES: dict[str, list[str]] = {
    JOY: [
        "这次{topic}体验让我很开心",
        "这个{topic}真的很精彩",
        "看到{topic}的表现我觉得很愉快",
        "这份{topic}体验带来了明显的快乐",
    ],
    SATISFACTION: [
        "这个{topic}使用起来很方便",
        "我对{topic}的整体表现比较满意",
        "{topic}的服务很贴心",
        "{topic}的质量很好，达到了预期",
    ],
    PRAISE: [
        "这个{topic}值得推荐给别人",
        "我想为这个{topic}点赞",
        "{topic}的表现非常优秀",
        "这个{topic}做得很专业，值得表扬",
        "{topic}的体验超出预期，值得称赞",
    ],
    DISAPPOINTMENT: [
        "这个{topic}让我很失望",
        "{topic}的表现没有达到预期",
        "这次{topic}体验比较一般",
        "{topic}的效果不太满意",
    ],
    ANGER: [
        "这个{topic}的表现太差了",
        "{topic}的问题让我很生气",
        "这次{topic}体验非常糟糕",
        "{topic}的服务差劲到让人愤怒",
    ],
    DISGUST: [
        "这个{topic}真的很无聊",
        "我很讨厌这次{topic}体验",
        "{topic}的内容让人反感",
        "这个{topic}给人的感觉很厌恶",
    ],
}


def infer_emotion_label(example: Example) -> str:
    text = example.text
    rules = POSITIVE_RULES if example.label == POSITIVE else NEGATIVE_RULES
    for emotion, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return emotion
    return SATISFACTION if example.label == POSITIVE else DISAPPOINTMENT


def build_multiclass_emotion_dataset(
    binary_examples: list[Example],
    neutral_per_topic: int = 8,
    min_examples_per_label: int = 80,
) -> list[EmotionExample]:
    examples = [
        EmotionExample(
            text=example.text,
            label=infer_emotion_label(example),
            polarity=example.label,
            source="教材数据弱标注",
        )
        for example in binary_examples
    ]
    examples.extend(generate_neutral_examples(neutral_per_topic=neutral_per_topic))
    examples.extend(generate_balancing_examples(examples, min_examples_per_label))
    return examples


def load_multiclass_emotion_dataset(data_dir: Path) -> list[EmotionExample]:
    return build_multiclass_emotion_dataset(load_sentiment_csv(data_dir / "sentiment.csv"))


def generate_neutral_examples(neutral_per_topic: int = 8) -> list[EmotionExample]:
    neutral_examples: list[EmotionExample] = []
    patterns = NEUTRAL_PATTERNS[:neutral_per_topic]
    for topic in NEUTRAL_TOPICS:
        for pattern in patterns:
            neutral_examples.append(
                EmotionExample(
                    text=pattern.format(topic=topic),
                    label=NEUTRAL,
                    polarity=NEUTRAL,
                    source="中性模板补充",
                )
            )
    return neutral_examples


def generate_balancing_examples(
    examples: list[EmotionExample],
    min_examples_per_label: int,
) -> list[EmotionExample]:
    counts = Counter(example.label for example in examples)
    augmented: list[EmotionExample] = []
    for label in EMOTION_LABELS:
        needed = max(0, min_examples_per_label - counts.get(label, 0))
        if needed == 0:
            continue
        if label == NEUTRAL:
            pool = generate_neutral_examples(neutral_per_topic=len(NEUTRAL_PATTERNS))
            templates = [example.text for example in pool]
        else:
            templates = [
                template.format(topic=topic)
                for template in EMOTION_AUGMENT_TEMPLATES[label]
                for topic in NEUTRAL_TOPICS
            ]
        for i in range(needed):
            text = templates[i % len(templates)]
            if i >= len(templates):
                text = f"{text}，样本{i + 1}"
            augmented.append(
                EmotionExample(
                    text=text,
                    label=label,
                    polarity=NEUTRAL if label == NEUTRAL else ("正面" if label in {JOY, SATISFACTION, PRAISE} else "负面"),
                    source="情绪模板增强",
                )
            )
    return augmented


def emotion_distribution(examples: list[EmotionExample]) -> dict[str, int]:
    counts = Counter(example.label for example in examples)
    return {label: counts.get(label, 0) for label in EMOTION_LABELS}


def explain_labeling_rules() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, keywords in POSITIVE_RULES + NEGATIVE_RULES:
        rows.append({"label": label, "keywords": "、".join(keywords)})
    rows.append({"label": NEUTRAL, "keywords": "客观陈述模板，无明显情感词"})
    return rows
