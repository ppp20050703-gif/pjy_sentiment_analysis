from __future__ import annotations

import csv
import json
from pathlib import Path


def generate_report_markdown(output_dir: Path, report_path: Path) -> None:
    summary_path = output_dir / "experiment_summary.json"
    metrics_path = output_dir / "metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("请先运行实验，生成 experiment_summary.json 和 metrics.csv")

    binary_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    binary_metrics = read_metrics(metrics_path)

    advanced_summary_path = output_dir / "advanced_multiclass" / "multiclass_summary.json"
    advanced_metrics_path = output_dir / "advanced_multiclass" / "multiclass_metrics.csv"
    sklearn_summary_path = output_dir / "sklearn_multiclass" / "sklearn_multiclass_summary.json"
    sklearn_status_path = output_dir / "sklearn_multiclass" / "sklearn_dependency_status.json"

    advanced_summary = read_json_if_exists(advanced_summary_path)
    advanced_metrics = read_metrics(advanced_metrics_path) if advanced_metrics_path.exists() else []
    sklearn_summary = read_json_if_exists(sklearn_summary_path)
    sklearn_status = read_json_if_exists(sklearn_status_path)

    lines: list[str] = [
        "# 文本情感分析系统作品设计报告材料",
        "",
        "## 选题与要求对应关系",
        "",
        "| 结课报告要求 | 本系统对应内容 |",
        "| --- | --- |",
        "| 至少对应教材中的一个具体任务或算法 | 对应教材第9章“文本情感分析”：情感词典计分、文本向量化、朴素贝叶斯情感分类。 |",
        "| 至少完成一个完整实验流程 | 已实现数据读取、弱标注扩展、预处理、训练/预测、测试集评估、5折交叉验证、混淆矩阵、实验结果导出。 |",
        "| 至少进行一种对比实验 | 二分类对比多数类基线、教材词典法、BoW+朴素贝叶斯、TF-IDF+朴素贝叶斯、TF-IDF+MaxEnt；多类别对比规则模型、BoW/TF-IDF朴素贝叶斯，并预留 scikit-learn 模型组。 |",
        "| 第一部分：教材程序复现与实现逻辑论述 | 复现教材第9章词典法与机器学习分类流程，并把原二分类任务扩展为细粒度情绪标签识别。 |",
        "| 第二部分：选题方向论文复现或实现逻辑论述 | 采用 Pang、Lee 与 Vaithyanathan（2002）情感分类机器学习路线，复现其“文本特征 + 分类模型对比”的核心实验思想；高级版支持 scikit-learn 的 MultinomialNB、LinearSVC、LogisticRegression、SGDClassifier、RandomForest。 |",
        "| 阐述原则 | 下文按 5W2H 组织。 |",
        "",
        "## 二分类教材复现实验",
        "",
        f"- 数据集：`{binary_summary['data_path']}`",
        f"- 样本数：{binary_summary['dataset_size']}，标签分布：{binary_summary['label_distribution']}",
        f"- 训练集/测试集：{binary_summary['train_size']} / {binary_summary['test_size']}",
        f"- 分词与特征：`{binary_summary['tokenizer']}`，最大特征数：{binary_summary['max_features']}",
        f"- 测试集最佳模型：{binary_summary['best_model']}",
        "",
        "| 模型 | 测试集Accuracy | 测试集Macro-F1 | 5折CV Accuracy | 5折CV Macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    append_metric_rows(lines, binary_metrics)

    if advanced_summary is not None:
        lines.extend(
            [
                "",
                "## 多类别细粒度情绪分类扩展",
                "",
                f"- 情绪标签：{'、'.join(advanced_summary['labels'])}",
                f"- 多类别样本数：{advanced_summary['dataset_size']}，标签分布：{advanced_summary['label_distribution']}",
                f"- 数据构造：{advanced_summary['labeling_method']}",
                f"- 多类别最佳模型：{advanced_summary['best_model']}",
                "",
                "| 模型 | 测试集Accuracy | 测试集Macro-F1 | 5折CV Accuracy | 5折CV Macro-F1 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        append_metric_rows(lines, advanced_metrics)

    lines.extend(["", "## scikit-learn模型组"])
    if sklearn_summary is not None:
        lines.append(
            f"已启用 scikit-learn 高级实验，最佳模型为 `{sklearn_summary['best_model_key']}`，"
            f"测试集 Macro-F1 为 {sklearn_summary['best_macro_f1']:.4f}。详细结果见 "
            "`outputs/sklearn_multiclass/sklearn_multiclass_metrics.csv`。"
        )
    elif sklearn_status is not None:
        lines.append(f"当前环境未启用 scikit-learn 实验：{sklearn_status['reason']}")
    else:
        lines.append("尚未运行 scikit-learn 实验。")

    lines.extend(
        [
            "",
            "## 第一部分：教材程序复现与实现逻辑论述（5W2H）",
            "",
            "### What（做什么）",
            "本部分完成教材第9章“文本情感分析”任务复现。基础输入为中文评论文本，基础输出为二分类情感标签：正面或负面；高级扩展输出为七类细粒度情绪标签：喜悦、满意、赞赏、失望、愤怒、厌恶、中性。复现对象包括两类教材算法：一是基于情感词典、否定词和程度副词的规则计分法；二是基于文本向量化和朴素贝叶斯分类器的机器学习方法。",
            "",
            "### Why（为什么做）",
            "情感分析是文本分类在主观性文本上的典型应用，能把评论、舆情、产品反馈等非结构化文本转化为可统计的情绪倾向。二分类能回答态度方向，多类别情绪能进一步回答用户具体是在高兴、满意、失望还是愤怒。该任务与教材第9章直接对应，同时能自然形成“规则方法 vs 机器学习方法”“BoW vs TF-IDF”“传统实现 vs scikit-learn库模型”的对比实验。",
            "",
            "### Who（谁参与/面向谁）",
            "系统面向课程作品设计报告的评阅场景，使用者为学生本人。系统输出的指标、预测文件和混淆矩阵可直接作为报告中的实验依据。",
            "",
            "### When（何时执行）",
            "实验按固定流程执行：准备教材数据集，进行分层训练/测试划分，训练多个模型，在同一测试集上评估，再进行5折交叉验证验证稳定性。",
            "",
            "### Where（在哪里实现）",
            "工程位于 `homework/sentiment_analysis_system`，教材数据来自 `文本情感分析教材代码/data/sentiment.csv`，输出结果位于 `outputs` 目录。",
            "",
            "### How（如何实现）",
            "数据处理阶段读取 UTF-8 CSV，并统一正面/负面标签；多类别阶段根据情绪关键词把教材样本弱标注为喜悦、满意、赞赏、失望、愤怒、厌恶，并用客观陈述模板补充中性类；预处理阶段采用中文字符 unigram + bigram，避免当前环境缺少 jieba 时无法运行，同时保留“不好、很棒、糟糕”等关键情感片段；建模阶段实现情感词典计分、BoW+朴素贝叶斯、TF-IDF+朴素贝叶斯，并支持 scikit-learn 管道模型；评估阶段输出 Accuracy、Precision、Recall、Macro-F1、混淆矩阵和预测明细。",
            "",
            "### How much（做到什么程度/成本与结果）",
            "本系统基础实验使用1000条教材样本，正负样本各500条。高级实验在此基础上构造七类情绪标签数据集。测试集比例为20%，交叉验证为5折。具体指标见上方实验结果表，详细分类报告见 `outputs/classification_reports.txt` 与 `outputs/advanced_multiclass/multiclass_classification_reports.txt`。",
            "",
            "## 第二部分：论文方向复现或实现逻辑论述（5W2H）",
            "",
            "### What（做什么）",
            "论文方向选择经典情感分类论文 Pang, Lee & Vaithyanathan (2002) 的机器学习实验路线。该路线将情感分析视为监督式文本分类问题，并比较不同特征与分类器。本系统实现其中的 MaxEnt 思路，即用 Logistic Regression 作为最大熵分类器，并与教材朴素贝叶斯方法进行对比；高级版进一步支持 scikit-learn 的多分类模型组。",
            "",
            "### Why（为什么选择该论文方向）",
            "该论文与教材第9章机器学习情感分析高度一致：都是将评论文本向量化后训练分类器。相比直接复现大规模深度模型，这一选择更符合课程报告对“论述逻辑与实现完整性”的要求，也能在普通电脑上稳定复现实验。",
            "",
            "### Who（谁提出/谁使用）",
            "Pang、Lee 与 Vaithyanathan 提出了早期电影评论情感分类的机器学习对比实验。本系统将其核心思想用于中文教材评论数据，使用者可以通过命令行复现实验并对新文本进行二分类或七分类情绪预测。",
            "",
            "### When（何时应用）",
            "当已有标注语料且需要可解释、可快速训练的情感分类模型时，MaxEnt/Logistic、朴素贝叶斯、线性SVM、随机森林适合作为基线和对比模型。本系统在教材复现实验之后加入该论文方向模型。",
            "",
            "### Where（在哪里落地）",
            "论文方向模型在 `sentiment_system/models.py` 的 `LogisticRegressionClassifier` 中实现；多类别扩展在 `sentiment_system/advanced_experiment.py` 中实现；scikit-learn 模型组在 `sentiment_system/sklearn_experiment.py` 中实现。",
            "",
            "### How（如何实现）",
            "系统先把中文评论转为字符 unigram/bigram，再用 TF-IDF 计算稀疏文本特征；二分类 Logistic 模型用 sigmoid 函数输出正面概率，用随机梯度下降优化交叉熵损失，并加入 L2 正则抑制过拟合。高级版在安装 scikit-learn 后可进一步调用 MultinomialNB、LinearSVC、LogisticRegression、SGDClassifier 和 RandomForestClassifier，形成更完整的库模型对比。该流程对应论文中的“文本特征 + 监督分类器 + 指标对比”的实验逻辑。",
            "",
            "### How much（效果如何）",
            "效果以 `TF-IDF + MaxEnt(Logistic)` 的二分类 Macro-F1，以及多类别情绪分类的 Macro-F1 为准。若库模型优于或接近手写朴素贝叶斯，可说明成熟机器学习库在优化器、正则化和模型接口上更完整；若不优于，也可从弱标注噪声、样本规模、特征粒度和语料模板化程度解释。",
            "",
            "## 可写入正文的结论",
            "",
            "1. 教材词典法不需要训练，解释性较强，但受词典覆盖率和否定/程度词窗口影响明显。",
            "2. 朴素贝叶斯完成了从文本向量化到分类评估的完整监督学习流程，是教材程序的核心复现。",
            "3. 七类情绪标签比正/负二分类更细，能支持喜悦、满意、赞赏、失望、愤怒、厌恶、中性等更丰富的语言情绪分析。",
            "4. TF-IDF 与 BoW 的对比体现了特征权重改进；MaxEnt(Logistic)、SVM、随机森林等模型与朴素贝叶斯的对比体现了分类模型改进。",
            "5. 本系统比教材零散示例更完整：补充了分层划分、交叉验证、统一指标、混淆矩阵、弱监督多类标签构造和可复现实验输出。",
            "",
            "## 参考文献建议",
            "",
            "[1] Pang B, Lee L, Vaithyanathan S. Thumbs up? Sentiment Classification using Machine Learning Techniques. EMNLP, 2002.",
            "[2] 教材第9章：文本情感分析相关程序，包含情感词典法、朴素贝叶斯分类、TF-IDF 与 SVM 示例。",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def append_metric_rows(lines: list[str], rows: list[dict[str, str]]) -> None:
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['accuracy']} | {row['macro_f1']} | "
            f"{row['cv_accuracy_mean']}±{row['cv_accuracy_std']} | "
            f"{row['cv_macro_f1_mean']}±{row['cv_macro_f1_std']} |"
        )


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
