#!/bin/bash

# AlexNet微调Caltech-101实验脚本

# 确保存在必要的目录
mkdir -p models results runs experiments

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${YELLOW}AlexNet微调Caltech-101实验脚本${NC}"
    echo "用法: ./run_experiment.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --test           : 测试环境和基本功能"
    echo "  --finetune       : 运行微调实验"
    echo "  --scratch        : 运行从零训练实验"
    echo "  --feature-extract: 运行特征提取实验(只训练最后一层)"
    echo "  --all            : 运行所有实验"
    echo "  --help           : 显示帮助信息"
    echo ""
    echo "参数:"
    echo "  --epochs N       : 设置训练轮数 (默认: 20)"
    echo "  --batch-size N   : 设置批处理大小 (默认: 32)"
    echo "  --lr N           : 设置学习率 (默认: 0.001)"
    echo "  --finetune-lr N  : 设置微调学习率 (默认: 0.0001)"
    echo ""
    echo "示例:"
    echo "  ./run_experiment.sh --test"
    echo "  ./run_experiment.sh --finetune --epochs 10 --batch-size 64"
    echo "  ./run_experiment.sh --all --epochs 5"
}

# 默认参数
EPOCHS=20
BATCH_SIZE=32
LR=0.001
FINETUNE_LR=0.0001

# 解析命令行参数
RUN_TEST=false
RUN_FINETUNE=false
RUN_SCRATCH=false
RUN_FEATURE_EXTRACT=false
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            RUN_TEST=true
            shift
            ;;
        --finetune)
            RUN_FINETUNE=true
            shift
            ;;
        --scratch)
            RUN_SCRATCH=true
            shift
            ;;
        --feature-extract)
            RUN_FEATURE_EXTRACT=true
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --epochs)
            EPOCHS=$2
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=$2
            shift 2
            ;;
        --lr)
            LR=$2
            shift 2
            ;;
        --finetune-lr)
            FINETUNE_LR=$2
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 如果没有指定任何操作，显示帮助信息
if [ "$RUN_TEST" = false ] && [ "$RUN_FINETUNE" = false ] && [ "$RUN_SCRATCH" = false ] && [ "$RUN_FEATURE_EXTRACT" = false ] && [ "$RUN_ALL" = false ]; then
    show_help
    exit 0
fi

# 运行环境测试
if [ "$RUN_TEST" = true ]; then
    echo -e "${GREEN}运行环境测试...${NC}"
    python test_environment.py
fi

# 运行微调实验
if [ "$RUN_FINETUNE" = true ] || [ "$RUN_ALL" = true ]; then
    echo -e "${GREEN}运行微调实验...${NC}"
    python main.py --mode finetune --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --finetune_lr $FINETUNE_LR
fi

# 运行从零训练实验
if [ "$RUN_SCRATCH" = true ] || [ "$RUN_ALL" = true ]; then
    echo -e "${GREEN}运行从零训练实验...${NC}"
    python main.py --mode scratch --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR
fi

# 运行特征提取实验
if [ "$RUN_FEATURE_EXTRACT" = true ] || [ "$RUN_ALL" = true ]; then
    echo -e "${GREEN}运行特征提取实验...${NC}"
    python main.py --mode finetune --feature_extract --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR
fi

echo -e "${GREEN}所有实验完成!${NC}"
echo "可以使用以下命令查看训练过程和结果:"
echo "tensorboard --logdir=runs" 