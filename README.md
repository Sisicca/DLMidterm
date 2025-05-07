# AlexNet 微调 - Caltech-101分类

此项目为神经网络课程作业，使用在ImageNet上预训练的AlexNet模型微调，实现Caltech-101数据集的图像分类任务。

## 项目概述

- 基于PyTorch实现的AlexNet微调项目
- 使用预训练的AlexNet模型在Caltech-101数据集上进行微调
- 与从头训练的模型进行对比，验证微调的有效性
- 提供TensorBoard可视化支持，记录训练损失和准确率

## 项目结构

- `data_utils.py`: 数据集加载和预处理，包括数据增强和标准化
- `model.py`: AlexNet模型定义，支持预训练和从零训练
- `train.py`: 训练和验证函数
- `evaluate.py`: 模型评估和性能分析
- `visualize.py`: 可视化工具，包括TensorBoard集成
- `main.py`: 主程序入口，整合所有模块

## 环境要求

- Python 3.6+
- PyTorch 2.0.0+
- torchvision 0.15.0+
- 其他依赖见requirements.txt

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集

项目使用Caltech-101数据集，已经下载并解压到`caltech-101`目录。

## 运行说明

### 微调预训练模型

```bash
python main.py --mode finetune --epochs 20 --batch_size 32 --lr 0.001 --finetune_lr 0.0001
```

### 从零开始训练模型

```bash
python main.py --mode scratch --epochs 20 --batch_size 32 --lr 0.001
```

### 仅训练最后一层（特征提取）

```bash
python main.py --mode finetune --feature_extract --epochs 20 --batch_size 32 --lr 0.001
```

### 评估模型

```bash
python main.py --mode evaluate --model_path models/finetune_xxxx/finetune_best.pth
```

## 超参数调优

可以调整以下超参数以获得更好的性能：

- `--batch_size`: 批处理大小，影响内存使用和训练速度
- `--epochs`: 训练轮数，更多轮数可能获得更好性能
- `--lr`: 学习率，控制参数更新步长
- `--finetune_lr`: 微调学习率，控制预训练层的更新步长
- `--feature_extract`: 是否只训练最后一层，可加快训练速度

## 结果可视化

训练过程中会自动记录以下信息到TensorBoard：

1. 训练和验证损失
2. 训练和验证准确率
3. 模型预测可视化

可以使用以下命令查看TensorBoard：

```bash
tensorboard --logdir=runs
```

## 模型保存

模型会保存在`models`目录下，包括：

- 每个epoch的中间模型
- 验证集上性能最好的模型
- 最终模型

## 结果分析

训练完成后，会在`results`目录下生成以下文件：

- 混淆矩阵
- 分类报告
- 模型预测可视化
- 训练信息汇总 