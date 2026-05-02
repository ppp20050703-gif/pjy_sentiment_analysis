# 文本情感分析系统

本工程用于《自然语言处理》结课大作业的“文本情感分析”作品设计部分，直接对应教材第9章代码与数据。

## 已满足的报告要求

- 教材任务/算法：第9章文本情感分析，包含情感词典法、文本向量化、朴素贝叶斯分类。
- 完整实验流程：数据读取、预处理、分层划分、训练、预测、测试集评估、5折交叉验证、结果导出。
- 对比实验：多数类基线、教材词典法、BoW+朴素贝叶斯、TF-IDF+朴素贝叶斯、TF-IDF+MaxEnt(Logistic)。
- 多类别扩展：在正/负二分类基础上扩展为 `喜悦、满意、赞赏、失望、愤怒、厌恶、中性` 七类细粒度情绪标签。
- 机器学习库扩展：安装依赖后可启用 `scikit-learn` 的 MultinomialNB、LinearSVC、LogisticRegression、SGDClassifier、RandomForestClassifier。
- 论文方向：按 Pang、Lee 与 Vaithyanathan（2002）情感分类机器学习路线，复现“文本特征 + 分类器对比”的实现逻辑。
- 报告组织：自动生成 `REPORT_5W2H.md`，按第一部分教材复现、第二部分论文方向和 5W2H 原则整理。

## 运行方式

在本目录执行：

```powershell
python main.py run
```

只运行多类别高级实验：

```powershell
python main.py advanced
```

如果默认 Python 无法输出中文，可使用：

```powershell
$env:PYTHONIOENCODING="utf-8"
python main.py run
```

预测新文本：

```powershell
python main.py predict --text "这个产品质量很好，使用起来非常方便" --text "服务态度太差了，不会再来了"
```

预测细粒度情绪标签：

```powershell
python main.py predict-emotion --text "这个项目值得推荐" --text "我只是查看了一下这家酒店的信息"
```

可选模型：

- `lexicon_textbook`
- `nb_bow_textbook`
- `nb_tfidf_improved`
- `maxent_tfidf_paper`

细粒度情绪模型：

- `emotion_rules`
- `emotion_nb_bow`
- `emotion_nb_tfidf`

## 启用 scikit-learn 模型组

基础系统无需第三方依赖即可运行。如果要启用机器学习库版高级实验，在本目录执行：

```powershell
python -m pip install --target vendor -r requirements.txt
python main.py advanced
```

安装成功后，系统会额外生成 `outputs/sklearn_multiclass/sklearn_multiclass_metrics.csv` 和 `best_sklearn_emotion_model.joblib`。

## 输出文件

- `outputs/metrics.csv`：各模型测试集与交叉验证指标。
- `outputs/classification_reports.txt`：每个模型的 Precision、Recall、F1。
- `outputs/predictions.csv`：测试集预测明细，可用于错误分析。
- `outputs/confusion_*.svg`：混淆矩阵图。
- `outputs/experiment_summary.json`：实验摘要。
- `outputs/advanced_multiclass/multiclass_emotion_dataset.csv`：七分类情绪数据集。
- `outputs/advanced_multiclass/multiclass_metrics.csv`：七分类模型对比指标。
- `outputs/advanced_multiclass/emotion_labeling_rules.csv`：弱标注规则说明。
- `outputs/sklearn_multiclass/`：安装 scikit-learn 后生成的库模型结果。
- `REPORT_5W2H.md`：可写入作品设计报告的材料。

## 说明

当前环境未安装 `jieba`、`scikit-learn`、`torch`，所以系统默认采用可直接运行的轻量实现：中文字符 unigram/bigram 分词、手写 TF-IDF、朴素贝叶斯和 Logistic Regression。这样可以保证作品在普通 Python 环境下可复现。若后续安装 `jieba`，可用 `--tokenizer jieba` 切换分词方式；若安装 `scikit-learn`，会自动启用更复杂的机器学习库模型组。
