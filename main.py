import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

from data_utils import load_caltech101_data, get_num_classes
from model import create_alexnet_model, get_parameter_groups
from train import train_model, validate_model
from evaluate import evaluate_model, plot_confusion_matrix, generate_classification_report, compare_models
from visualize import show_batch, log_model_graph, visualize_model_predictions, log_images_to_tensorboard

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='AlexNet微调 - Caltech101分类')
    parser.add_argument('--mode', type=str, default='finetune', choices=['finetune', 'scratch', 'evaluate'],
                        help='运行模式: finetune, scratch, evaluate')
    parser.add_argument('--data_dir', type=str, default='caltech-101/101_ObjectCategories',
                        help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--finetune_lr', type=float, default=0.0001,
                        help='微调学习率')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径，用于评估模式')
    parser.add_argument('--feature_extract', action='store_true',
                        help='仅训练最后一层')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 检查GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"加载Caltech-101数据集...")
    train_loader, val_loader, class_to_idx = load_caltech101_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # 获取类别数量和类别名称
    num_classes = get_num_classes(args.data_dir)
    print(f"类别数量: {num_classes}")
    
    # 创建类别名称映射
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'{args.mode}_{timestamp}')
    models_dir = os.path.join('models', f'{args.mode}_{timestamp}')
    log_dir = os.path.join('runs', f'{args.mode}_{timestamp}')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    if args.mode == 'finetune' or args.mode == 'scratch':
        # 创建模型
        pretrained = (args.mode == 'finetune')
        model = create_alexnet_model(num_classes=num_classes, pretrained=pretrained)
        model = model.to(device)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 配置优化器参数
        params_to_update = get_parameter_groups(
            model, 
            feature_extract=args.feature_extract,
            lr=args.lr,
            finetune_lr=args.finetune_lr
        )
        
        # 创建优化器
        optimizer = optim.Adam(params_to_update)
        
        # 显示数据样本
        show_batch(
            train_loader, 
            class_names=class_names,
            save_path=os.path.join(results_dir, 'data_samples.png')
        )
        
        # 记录模型图到TensorBoard
        # log_model_graph(model, train_loader, log_dir)
        
        # 训练模型
        print(f"开始训练模型...")
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=args.epochs,
            device=device,
            model_save_path=models_dir,
            log_dir=log_dir,
            model_name=args.mode
        )
        
        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(models_dir, f'{args.mode}_final.pth'))
        
        # 评估最终模型
        print(f"评估最终模型...")
        loss, accuracy, all_preds, all_labels = evaluate_model(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names
        )
        
        # 保存模型信息
        with open(os.path.join(results_dir, 'model_results.txt'), 'w') as f:
            f.write(f"模式: {args.mode}\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"损失: {loss:.4f}\n")
            f.write(f"类别数量: {num_classes}\n")
            f.write(f"批处理大小: {args.batch_size}\n")
            f.write(f"训练轮数: {args.epochs}\n")
            f.write(f"学习率: {args.lr}\n")
            f.write(f"微调学习率: {args.finetune_lr}\n")
            f.write(f"仅训练最后一层: {args.feature_extract}\n")
        
        # 绘制混淆矩阵
        print(f"绘制混淆矩阵...")
        plot_confusion_matrix(
            true_labels=all_labels,
            predictions=all_preds,
            class_names=class_names[:20],  # 只显示前20个类别
            save_path=os.path.join(results_dir, 'confusion_matrix.png')
        )
        
        # 生成分类报告
        print(f"生成分类报告...")
        generate_classification_report(
            true_labels=all_labels,
            predictions=all_preds,
            class_names=class_names,
            save_path=os.path.join(results_dir, 'classification_report.txt')
        )
        
        # 可视化模型预测
        print(f"可视化模型预测...")
        visualize_model_predictions(
            model=model,
            dataloader=val_loader,
            class_names=class_names,
            device=device,
            save_path=os.path.join(results_dir, 'model_predictions.png')
        )
        
    elif args.mode == 'evaluate':
        # 检查模型路径
        if args.model_path is None or not os.path.exists(args.model_path):
            print(f"错误: 请提供有效的模型路径")
            return
        
        # 创建模型
        model = create_alexnet_model(num_classes=num_classes)
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 评估模型
        print(f"评估模型: {args.model_path}")
        loss, accuracy, all_preds, all_labels = evaluate_model(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names
        )
        
        # 保存评估结果
        with open(os.path.join(results_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(f"模型路径: {args.model_path}\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"损失: {loss:.4f}\n")
        
        # 绘制混淆矩阵
        plot_confusion_matrix(
            true_labels=all_labels,
            predictions=all_preds,
            class_names=class_names[:20],
            save_path=os.path.join(results_dir, 'confusion_matrix.png')
        )
        
        # 生成分类报告
        generate_classification_report(
            true_labels=all_labels,
            predictions=all_preds,
            class_names=class_names,
            save_path=os.path.join(results_dir, 'classification_report.txt')
        )
    
    print(f"完成! 结果保存在: {results_dir}")

if __name__ == "__main__":
    main() 