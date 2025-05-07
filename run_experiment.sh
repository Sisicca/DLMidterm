#!/bin/bash

# 初始化Git仓库（如果不存在）
if [ ! -d .git ]; then
    git init
    git add .
    git commit -m "初始提交"
fi

# 下载数据集
echo "步骤1: 下载数据集"
python download_dataset.py

# 创建必要的目录
mkdir -p data
mkdir -p models
mkdir -p results
mkdir -p runs

# 运行超参数搜索
echo "步骤2: 超参数搜索（可选，耗时较长）"
read -p "是否运行超参数搜索？(y/n) " choice
if [ "$choice" = "y" ]; then
    python hyperparameter_search.py
    git add results/hyperparameter_search
    git commit -m "超参数搜索结果"
fi

# 微调模型
echo "步骤3: 训练微调模型"
python main.py --mode finetune --num_epochs 25 --learning_rate 0.001 --batch_size 32
git add models/ results/ runs/
git commit -m "微调模型训练结果"

# 从头训练模型
echo "步骤4: 从头训练模型"
python main.py --mode scratch --num_epochs 25 --learning_rate 0.01 --batch_size 32
git add models/ results/ runs/
git commit -m "从头训练模型训练结果"

# 比较两种模型
echo "步骤5: 比较两种模型"
python main.py --mode compare --model_path models/alexnet_finetune_best.pth --model_scratch_path models/alexnet_scratch_best.pth
git add results/
git commit -m "模型比较结果"

echo "实验完成！"
echo "结果保存在 results/ 目录下"
echo "训练日志可通过TensorBoard查看: tensorboard --logdir=runs" 