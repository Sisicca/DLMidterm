import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 检查导入的模块
try:
    from data_utils import load_caltech101_data, get_num_classes
    from model import create_alexnet_model, get_parameter_groups
    from train import train_model, validate_model
    from evaluate import evaluate_model
    from visualize import show_batch
    print("✓ 所有模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入错误: {e}")

def test_torch_availability():
    """测试PyTorch是否可用"""
    print("\n测试PyTorch环境:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"torchvision版本: {torchvision.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")

def test_data_loading():
    """测试数据加载功能"""
    print("\n测试数据加载:")
    try:
        data_dir = 'caltech-101/101_ObjectCategories'
        if not os.path.exists(data_dir):
            print(f"✗ 数据目录不存在: {data_dir}")
            return
        
        # 测试获取类别数量
        num_classes = get_num_classes(data_dir)
        print(f"类别数量: {num_classes}")
        
        # 测试加载少量数据
        batch_size = 4
        train_loader, val_loader, class_to_idx = load_caltech101_data(
            data_dir=data_dir,
            batch_size=batch_size,
            train_ratio=0.8
        )
        
        # 显示加载的批次信息
        print(f"训练数据加载器批次数: {len(train_loader)}")
        print(f"验证数据加载器批次数: {len(val_loader)}")
        print(f"类别到索引映射条目数: {len(class_to_idx)}")
        
        # 获取一个批次的数据并显示形状
        images, labels = next(iter(train_loader))
        print(f"图像批次形状: {images.shape}")
        print(f"标签批次形状: {labels.shape}")
        
        print("✓ 数据加载测试成功")
    except Exception as e:
        print(f"✗ 数据加载错误: {e}")

def test_model_creation():
    """测试模型创建功能"""
    print("\n测试模型创建:")
    try:
        # 创建预训练模型
        model_pretrained = create_alexnet_model(num_classes=101, pretrained=True)
        print("✓ 创建预训练模型成功")
        
        # 创建从零训练模型
        model_scratch = create_alexnet_model(num_classes=101, pretrained=False)
        print("✓ 创建从零训练模型成功")
        
        # 检查最后一层是否正确修改
        last_layer = model_pretrained.classifier[6]
        if isinstance(last_layer, torch.nn.Linear) and last_layer.out_features == 101:
            print("✓ 输出层维度正确")
        else:
            print(f"✗ 输出层维度错误: {last_layer.out_features}")
        
        # 测试参数分组功能
        params_finetune = get_parameter_groups(model_pretrained, feature_extract=False, lr=0.001, finetune_lr=0.0001)
        params_feature_extract = get_parameter_groups(model_pretrained, feature_extract=True, lr=0.001)
        
        print(f"✓ 微调参数组数量: {len(params_finetune)}")
        print(f"✓ 特征提取参数组数量: {len(params_feature_extract)}")
    except Exception as e:
        print(f"✗ 模型创建错误: {e}")

def test_directories():
    """测试目录结构"""
    print("\n测试项目目录:")
    dirs_to_check = ['models', 'results', 'runs']
    for dir_name in dirs_to_check:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"创建目录: {dir_name}")
        else:
            print(f"目录已存在: {dir_name}")

def main():
    print("=" * 50)
    print("AlexNet微调 - Caltech-101环境测试")
    print("=" * 50)
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {os.getcwd()}")
    
    # 运行测试
    test_torch_availability()
    test_directories()
    test_data_loading()
    test_model_creation()
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main() 