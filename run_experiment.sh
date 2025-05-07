#!/bin/bash

# 设置日期时间戳
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
echo "开始实验: $TIMESTAMP"

# 初始化Git仓库（如果不存在）
if [ ! -d .git ]; then
    echo "初始化Git仓库..."
    git init
    git add .
    git commit -m "初始提交"
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p data
mkdir -p models
mkdir -p results
mkdir -p runs

# 记录系统信息
echo "记录系统信息..."
echo "系统信息: $(uname -a)" > results/experiment_${TIMESTAMP}.log
echo "Python版本: $(python --version)" >> results/experiment_${TIMESTAMP}.log
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')" >> results/experiment_${TIMESTAMP}.log
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')" >> results/experiment_${TIMESTAMP}.log
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "CUDA版本: $(python -c 'import torch; print(torch.version.cuda)')" >> results/experiment_${TIMESTAMP}.log
    echo "GPU型号: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')" >> results/experiment_${TIMESTAMP}.log
fi

# 运行超参数搜索
echo "步骤1: 超参数搜索（可选，耗时较长）"
read -p "是否运行超参数搜索？(y/n) " choice
if [ "$choice" = "y" ]; then
    echo "开始超参数搜索..."
    python hyperparameter_search.py
    git add results/hyperparameter_search
    git commit -m "超参数搜索结果"
fi

# 微调模型
echo "步骤2: 训练微调模型"
python main.py --mode finetune --num_epochs 25 --learning_rate 0.001 --batch_size 32 --weight_decay 1e-4
echo "微调模型训练完成。"
git add models/ results/ runs/
git commit -m "微调模型训练结果"

# 从头训练模型
echo "步骤3: 从头训练模型"
python main.py --mode scratch --num_epochs 25 --learning_rate 0.01 --batch_size 32 --weight_decay 1e-4
echo "从头训练模型完成。"
git add models/ results/ runs/
git commit -m "从头训练模型训练结果"

# 比较两种模型
echo "步骤4: 比较两种模型"
python main.py --mode compare --model_path models/best_model.pth --model_scratch_path models/alexnet_scratch_best.pth
echo "模型比较完成。"
git add results/
git commit -m "模型比较结果"

# 生成可视化结果
echo "步骤5: 生成其他可视化结果"
python visualize.py --model_path models/best_model.pth
echo "可视化完成。"
git add results/
git commit -m "可视化结果"

echo "======================"
echo "实验完成！"
echo "结果保存在 results/ 目录下"
echo "训练日志可通过TensorBoard查看: tensorboard --logdir=runs"
echo "实验日期时间: $TIMESTAMP"
echo "======================" 