# Dataset

本项目使用 AI4I 2020 Predictive Maintenance Dataset。

- 数据来源：UCI Machine Learning Repository
- 数据页面：https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- 本仓库附带文件：`data/raw/ai4i2020.csv`
- 文件大小：约 522 KB
- 样本数：10000
- 任务：工业设备故障二分类预测

## 字段说明

主要输入特征包括：

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

目标字段：

- `Machine failure`

故障类型标签：

- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

实验中不会将故障类型标签作为模型输入，以避免数据泄漏；这些字段仅用于故障类型统计和错误分析。
