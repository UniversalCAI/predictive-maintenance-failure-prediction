# Physics-Informed Cost-Sensitive Failure Prediction

本仓库是《高级机器学习理论》课程报告的开源实验代码，主题为：

> 面向工业设备预测性维护的物理机理增强成本敏感故障预测方法研究

项目使用 AI4I 2020 Predictive Maintenance Dataset，围绕工业设备预测性维护中的类别不平衡、漏报成本高和模型可解释性问题，构建完整的机器学习实验流程。

## 主要内容

- 基础模型：Logistic Regression、KNN、Linear SVM、Decision Tree。
- 集成模型：Random Forest、XGBoost、LightGBM。
- 物理机理增强特征：温差、功率代理、转速-扭矩比、磨损-负载交互等。
- 成本敏感学习：类别权重、非对称损失、成本矩阵和阈值优化。
- 集成与异常检测：OOF Stacking、IsolationForest 异常检测辅助特征。
- 评价指标：Accuracy、Precision、Recall、F1、ROC-AUC、PR-AUC、MCC、Balanced Accuracy、维护成本。
- 可解释性：树模型特征重要性和 SHAP 分析。

## 目录结构

```text
.
├── data/raw/ai4i2020.csv          # 数据集，体积较小，已随仓库附带
├── src/run_all.py                 # 一键复现实验脚本
├── outputs/figures/               # 已生成的主要图表
├── outputs/tables/                # 已生成的实验结果表
├── report/course_report.pdf       # 课程报告 PDF
├── requirements.txt               # Python 依赖
├── DATASET.md                     # 数据集说明
└── .gitignore
```

## 运行环境

推荐 Python 3.10 及以上版本。原实验环境为 Python 3.13。

安装依赖：

```bash
python -m pip install -r requirements.txt
```

一键运行完整实验：

```bash
python src/run_all.py
```

脚本会自动创建或覆盖以下输出目录：

```text
outputs/figures/
outputs/tables/
outputs/models/
```

其中 `outputs/models/` 保存训练后的模型文件，默认不纳入 Git 跟踪。

## 数据说明

数据文件 `data/raw/ai4i2020.csv` 已包含在仓库中，大小约 522 KB。若删除该文件，脚本也会尝试从 UCI Machine Learning Repository 自动下载。

注意：`TWF`、`HDF`、`PWF`、`OSF`、`RNF` 是故障类型标签。实验中不会将这些字段作为模型输入特征，以避免数据泄漏；它们仅用于故障类型统计和错误分析。

## 复现实验结果摘要

成本敏感目标下的最佳模型：

- 模型：`XGBoost_Weighted_Physics`
- 成本优化阈值：`0.19`
- Accuracy：`0.909`
- Recall：`0.956`
- Balanced Accuracy：`0.932`
- ROC-AUC：`0.978`
- PR-AUC：`0.837`
- 混淆矩阵：TN=1753, FP=179, FN=3, TP=65
- 仿真维护成本：`3290`

统计指标平衡目标下的代表模型：

- 模型：`LightGBM_Physics`
- Accuracy：`0.992`
- Precision：`0.919`
- Recall：`0.838`
- F1：`0.877`
- MCC：`0.874`
- PR-AUC：`0.893`

## 主要输出文件

结果表：

- `outputs/tables/results_default_threshold.csv`
- `outputs/tables/results_cost_optimized.csv`
- `outputs/tables/results_f1_optimized.csv`
- `outputs/tables/results_mcc_optimized.csv`
- `outputs/tables/ablation_results.csv`
- `outputs/tables/best_model_summary.json`

图表：

- `outputs/figures/class_distribution.png`
- `outputs/figures/failure_type_distribution.png`
- `outputs/figures/roc_curves.png`
- `outputs/figures/pr_curves.png`
- `outputs/figures/threshold_sweep_best_model.png`
- `outputs/figures/best_model_confusion_matrix.png`
- `outputs/figures/feature_importance.png`
- `outputs/figures/shap_summary.png`

## 引用

数据集：

```text
Dua, Dheeru and Graff, Casey. UCI Machine Learning Repository:
AI4I 2020 Predictive Maintenance Dataset. 2020.
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
```

## License

代码按 MIT License 开源。数据集请遵循 UCI Machine Learning Repository 和原数据集页面的使用说明。
