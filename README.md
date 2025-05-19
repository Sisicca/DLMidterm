# AlexNet 微调 - Caltech-101分类

此项目为神经网络课程作业，使用在ImageNet上预训练的AlexNet模型微调，实现Caltech-101数据集的图像分类任务。

## 项目概述

- 基于PyTorch实现的AlexNet微调项目
- 使用预训练的AlexNet模型在Caltech-101数据集上进行微调
- 与从头训练的模型进行对比，验证微调的有效性
- 提供TensorBoard可视化支持，记录训练损失和准确率
- 支持多种优化器和学习率调度策略
- 提供高级可视化工具，如Grad-CAM热力图和特征图可视化
- 支持TensorBoard图表导出和错误分类分析

## 项目结构

- `data_utils.py`: 数据集加载和预处理，包括数据增强和标准化
- `model.py`: AlexNet模型定义，支持预训练和从零训练
- `train.py`: 训练和验证函数
- `evaluate.py`: 模型评估和性能分析，包括错误分类分析
- `visualize.py`: 基础可视化工具，包括TensorBoard集成
- `advanced_visualization.py`: 高级可视化工具，包括Grad-CAM和特征图可视化
- `visualize_model.py`: 模型可视化脚本
- `utils.py`: 工具函数，包括优化器和学习率调度器工厂
- `tensorboard_exporter.py`: TensorBoard图表导出工具
- `main.py`: 主程序入口，整合所有模块

## 环境管理

- 使用uv管理依赖
- 其他依赖见pyproject.toml

## 安装uv

```bash
pip install uv
```

## 安装依赖

```bash
uv venv -p 3.10
source .venv/bin/activate
uv run hello.py
```

## 数据集

项目使用Caltech-101数据集，已经下载并解压到`caltech-101`目录。
数据集文件夹结构如下：

```bash
caltech-101/
├── 101_ObjectCategories/
├── Annotations/
├── show_annotation.m
```
## 运行说明

### 微调预训练模型

```bash
# 基础微调命令
uv run main.py --mode finetune --epochs 20 --batch_size 32 --lr 0.001 --finetune_lr 0.0001

# 推荐的完整微调命令
uv run main.py --mode finetune --epochs 20 --batch_size 32 --lr 0.001 --finetune_lr 0.0001 --optimizer adamw --weight_decay 0.0001 --scheduler cosine_warmup --warmup_steps 500 --save_frequency 5 --mixed_precision --early_stopping 5 --grad_clip 1.0
```

### 从零开始训练模型

```bash
uv run main.py --mode scratch --epochs 20 --batch_size 32 --lr 0.001
```

### 仅训练最后一层（特征提取）

```bash
uv run main.py --mode finetune --feature_extract --epochs 20 --batch_size 32 --lr 0.001
```

### 评估模型

```bash
uv run main.py --mode evaluate --model_path models/finetune_xxxx/finetune_best.pth
```

### 启动TensorBoard或导出TensorBoard图表

```bash
# 启动TensorBoard服务
uv run main.py --mode tensorboard --tensorboard_dir runs/finetune_xxxx

# 导出TensorBoard图表为PNG格式
uv run main.py --mode tensorboard --tensorboard_dir runs/finetune_xxxx --export_dir results/exported_charts
```

## 优化器选择

项目支持多种优化器，可以通过`--optimizer`参数选择：

```bash
uv run main.py --mode finetune --optimizer adam --lr 0.001
```

支持的优化器包括：
- `sgd`: 随机梯度下降（默认带动量0.9）
- `adam`: Adam优化器
- `adamw`: AdamW优化器（带权重衰减）
- `rmsprop`: RMSprop优化器
- `adadelta`: Adadelta优化器
- `adagrad`: Adagrad优化器

## 学习率调度

项目支持多种学习率调度策略，可以通过`--scheduler`参数选择：

```bash
uv run main.py --mode finetune --scheduler cosine_warmup --warmup_steps 500 --min_lr 1e-6
```

支持的调度器包括：
- `constant`: 恒定学习率
- `warmup`: 预热阶段
- `linear_decay`: 线性衰减
- `linear_warmup_decay`: 预热后线性衰减
- `cosine`: 余弦衰减
- `cosine_warmup`: 预热后余弦衰减
- `step`: 步进式衰减
- `reduce_on_plateau`: 当指标停止改善时降低学习率

## 高级训练选项

```bash
# 使用梯度裁剪
uv run main.py --mode finetune --grad_clip 1.0

# 使用早停
uv run main.py --mode finetune --early_stopping 5

# 使用混合精度训练(可加速训练过程)
uv run main.py --mode finetune --mixed_precision
```

## 可视化配置选项

```bash
# 自定义混淆矩阵显示
uv run main.py --mode finetune --normalize_cm --max_classes 30

# 不使用归一化混淆矩阵
uv run main.py --mode finetune --normalize_cm=False

# 修改默认TensorBoard端口
uv run main.py --mode tensorboard --tensorboard_dir runs/finetune_xxxx --tensorboard_port 8080
```

## 高级可视化

项目提供了高级可视化工具，帮助理解模型：

```bash
# 可视化单张图像的Grad-CAM热力图
uv run visualize_model.py --model_path models/finetune_xxxx/finetune_best.pth --image_path path/to/image.jpg --mode gradcam

# 可视化模型特征图
uv run visualize_model.py --model_path models/finetune_xxxx/finetune_best.pth --mode features --target_layer features.10

# 可视化数据集中多张图像的模型激活
uv run visualize_model.py --model_path models/finetune_xxxx/finetune_best.pth --mode activations --num_images 4
```

## 结果可视化

训练过程中会自动记录以下信息到TensorBoard：

1. 训练和验证损失
2. 训练和验证准确率
3. 学习率变化
4. 梯度范数
5. 模型预测可视化

可以使用以下命令查看TensorBoard：

```bash
tensorboard --logdir=runs
```

或者使用内置的TensorBoard模式：

```bash
uv run main.py --mode tensorboard --tensorboard_dir runs/finetune_xxxx
```

## 模型保存

模型会保存在`models`目录下，包括：

- 验证集上性能最好的模型（始终保存）
- 每隔一定epoch保存的检查点模型（可通过`--save_frequency`参数控制）
- 最终模型

默认情况下，每个epoch都会保存一个检查点（`--save_frequency 1`），这可能会占用大量磁盘空间。您可以通过设置`--save_frequency`参数来控制保存频率，例如`--save_frequency 5`表示每5个epoch保存一次模型。

## 结果分析

训练完成后，会在`results`目录下生成以下文件：

- 混淆矩阵（支持归一化和类别数量限制）
- 分类报告（详细的性能指标）
- 模型预测可视化
- 错误分类分析（展示最常见的错误分类对）
- 训练信息汇总
- TensorBoard图表导出（如学习率变化、损失曲线等）