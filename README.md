# AlexNet 微调 - Caltech-101分类

此项目为神经网络课程作业，使用在ImageNet上预训练的AlexNet模型微调，实现Caltech-101数据集的图像分类任务。

## 项目结构
- `data_utils.py`: 数据集加载和预处理
- `model.py`: 模型定义
- `train.py`: 训练脚本
- `evaluate.py`: 评估脚本
- `visualize.py`: 可视化工具
- `main.py`: 主程序入口

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集

项目使用[Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02)数据集，需要手动下载并放置在`data`目录下。

## 运行

```bash
# 训练微调模型
python main.py --mode finetune

# 训练从零开始的模型
python main.py --mode scratch

# 评估模型
python main.py --mode evaluate --model_path path/to/model.pth
```

## 超参数调优

脚本支持多种超参数调整，包括：
- 学习率
- 批次大小
- 训练步数
- 微调策略 