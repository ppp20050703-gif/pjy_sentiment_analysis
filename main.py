from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_system.advanced_experiment import (
    run_multiclass_experiments,
    train_multiclass_predictor,
)
from sentiment_system.data import find_textbook_data_dir, load_sentiment_csv
from sentiment_system.experiment import make_config, run_experiments, train_for_prediction
from sentiment_system.report import generate_report_markdown
from sentiment_system.sklearn_experiment import run_sklearn_multiclass_experiments


PROJECT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中文文本情感分析课程作品系统")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="运行完整实验流程并生成报告材料")
    add_common_args(run_parser)

    advanced_parser = subparsers.add_parser("advanced", help="运行多类别细粒度情绪分类实验")
    add_common_args(advanced_parser)

    report_parser = subparsers.add_parser("report", help="根据已有实验输出重新生成报告材料")
    report_parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs"))
    report_parser.add_argument("--report-path", default=str(PROJECT_DIR / "REPORT_5W2H.md"))

    predict_parser = subparsers.add_parser("predict", help="训练指定模型并预测新文本")
    add_common_args(predict_parser)
    predict_parser.add_argument(
        "--model",
        default="maxent_tfidf_paper",
        help="模型key，如 lexicon_textbook、nb_bow_textbook、nb_tfidf_improved、maxent_tfidf_paper",
    )
    predict_parser.add_argument("--text", action="append", help="待预测文本，可多次传入")
    predict_parser.add_argument("--input-file", help="每行一条待预测文本")

    emotion_parser = subparsers.add_parser("predict-emotion", help="预测细粒度情绪标签")
    add_common_args(emotion_parser)
    emotion_parser.add_argument(
        "--model",
        default="emotion_nb_tfidf",
        help="模型key，如 emotion_rules、emotion_nb_bow、emotion_nb_tfidf",
    )
    emotion_parser.add_argument("--text", action="append", help="待预测文本，可多次传入")
    emotion_parser.add_argument("--input-file", help="每行一条待预测文本")
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=None, help="教材 data 目录，默认自动查找")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs"), help="实验输出目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument(
        "--tokenizer",
        default="char_bigram",
        choices=["char", "char_bigram", "jieba"],
        help="分词方式；jieba 未安装时会自动回退到 char_bigram",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "run"

    if command == "run":
        config = make_config(
            args.data_dir,
            args.output_dir,
            args.seed,
            args.test_size,
            args.cv,
            args.max_features,
            args.tokenizer,
        )
        summary = run_experiments(config)
        advanced_summary = run_multiclass_experiments(config)
        sklearn_summary = run_sklearn_multiclass_experiments(config)
        report_path = PROJECT_DIR / "REPORT_5W2H.md"
        generate_report_markdown(config.output_dir, report_path)
        print("实验完成")
        print(f"数据集: {summary['dataset_size']} 条")
        print(f"二分类最佳模型: {summary['best_model']}")
        print(f"多类别最佳模型: {advanced_summary['best_model']}")
        if sklearn_summary.get("enabled"):
            print(f"sklearn多类别最佳Macro-F1: {sklearn_summary['best_macro_f1']:.4f}")
        else:
            print("sklearn多类别实验未启用: 当前环境缺少 scikit-learn")
        print(f"指标文件: {config.output_dir / 'metrics.csv'}")
        print(f"多类别指标: {config.output_dir / 'advanced_multiclass' / 'multiclass_metrics.csv'}")
        print(f"报告材料: {report_path}")
        return

    if command == "advanced":
        config = make_config(
            args.data_dir,
            args.output_dir,
            args.seed,
            args.test_size,
            args.cv,
            args.max_features,
            args.tokenizer,
        )
        advanced_summary = run_multiclass_experiments(config)
        sklearn_summary = run_sklearn_multiclass_experiments(config)
        generate_report_markdown(config.output_dir, PROJECT_DIR / "REPORT_5W2H.md")
        print("多类别情绪实验完成")
        print(f"情绪标签: {'、'.join(advanced_summary['labels'])}")
        print(f"样本数: {advanced_summary['dataset_size']}")
        print(f"多类别最佳模型: {advanced_summary['best_model']}")
        if sklearn_summary.get("enabled"):
            print(f"sklearn多类别最佳Macro-F1: {sklearn_summary['best_macro_f1']:.4f}")
        else:
            print("sklearn多类别实验未启用: 当前环境缺少 scikit-learn")
        return

    if command == "report":
        output_dir = Path(args.output_dir)
        report_path = Path(args.report_path)
        generate_report_markdown(output_dir, report_path)
        print(f"报告材料已生成: {report_path}")
        return

    if command == "predict":
        config = make_config(
            args.data_dir,
            args.output_dir,
            args.seed,
            args.test_size,
            args.cv,
            args.max_features,
            args.tokenizer,
        )
        texts = list(args.text or [])
        if args.input_file:
            texts.extend(
                line.strip()
                for line in Path(args.input_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if not texts:
            texts = ["这个产品质量很好，使用起来非常方便", "服务态度太差了，不会再来了"]
        data_dir = config.data_dir if config.data_dir else find_textbook_data_dir()
        examples = load_sentiment_csv(data_dir / "sentiment.csv")
        predict = train_for_prediction(args.model, examples, config)
        for text, label in zip(texts, predict(texts)):
            print(f"{label}\t{text}")
        return

    if command == "predict-emotion":
        config = make_config(
            args.data_dir,
            args.output_dir,
            args.seed,
            args.test_size,
            args.cv,
            args.max_features,
            args.tokenizer,
        )
        texts = list(args.text or [])
        if args.input_file:
            texts.extend(
                line.strip()
                for line in Path(args.input_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if not texts:
            texts = [
                "这个产品质量很好，使用起来非常方便",
                "服务态度太差了，不会再来了",
                "我只是查看了一下这家酒店的信息",
                "这部电影太无聊了",
            ]
        predict = train_multiclass_predictor(args.model, config)
        for text, label in zip(texts, predict(texts)):
            print(f"{label}\t{text}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
