import os
import argparse
import torch
import json
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_utils import get_data_loaders
from model import create_alexnet_model
from train import fine_tune_alexnet, train_alexnet_from_scratch
from evaluate import evaluate_model, plot_confusion_matrix, generate_classification_report, load_model
from visualize import plot_learning_curves, visualize_model_predictions, compare_learning_curves, visualize_feature_maps

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='AlexNet微调 - Caltech-101')
    parser.add_argument('--data_dir', type=str, default='data/caltech101',
                        help='数据集根目录')
    parser.add_argument('--mode', type=str, default='finetune',
                        choices=['finetune', 'scratch', 'evaluate', 'compare'],
                        help='运行模式: finetune, scratch, evaluate, compare')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='基础学习率')
    parser.add_argument('--feature_extract', action='store_true',
                        help='是否只训练最后一层')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径，用于评估或比较')
    parser.add_argument('--model_scratch_path', type=str, default=None,
                        help='从头训练的模型路径，用于比较')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 加载数据
    train_loader, test_loader, num_classes = get_data_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }
    
    # 获取类别名称
    dataset = train_loader.dataset
    class_names = [dataset.classes[i] for i in range(len(dataset.classes))]
    
    if args.mode == 'finetune':
        # 微调模型
        print("微调AlexNet模型...")
        model = create_alexnet_model(num_classes=num_classes, pretrained=True)
        model = model.to(device)
        
        model, history = fine_tune_alexnet(
            model, dataloaders, device,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            feature_extract=args.feature_extract
        )
        
        # 保存训练历史
        with open(f'results/finetune_history.json', 'w') as f:
            json.dump(history, f)
        
        # 可视化训练过程
        plot_learning_curves(history, save_path='results/finetune_learning_curves.png')
        
        # 评估模型
        criterion = nn.CrossEntropyLoss()
        acc, loss, preds, labels = evaluate_model(model, test_loader, device, criterion)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(labels, preds, class_names, save_path='results/finetune_confusion_matrix.png')
        
        # 生成分类报告
        generate_classification_report(labels, preds, class_names, save_path='results/finetune_classification_report.txt')
        
        # 可视化模型预测结果
        visualize_model_predictions(model, test_loader, device, class_names, save_path='results/finetune_predictions.png')
        
    elif args.mode == 'scratch':
        # 从头训练模型
        print("从头训练AlexNet模型...")
        model = create_alexnet_model(num_classes=num_classes, pretrained=False)
        model = model.to(device)
        
        model, history = train_alexnet_from_scratch(
            model, dataloaders, device,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs
        )
        
        # 保存训练历史
        with open(f'results/scratch_history.json', 'w') as f:
            json.dump(history, f)
        
        # 可视化训练过程
        plot_learning_curves(history, save_path='results/scratch_learning_curves.png')
        
        # 评估模型
        criterion = nn.CrossEntropyLoss()
        acc, loss, preds, labels = evaluate_model(model, test_loader, device, criterion)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(labels, preds, class_names, save_path='results/scratch_confusion_matrix.png')
        
        # 生成分类报告
        generate_classification_report(labels, preds, class_names, save_path='results/scratch_classification_report.txt')
        
        # 可视化模型预测结果
        visualize_model_predictions(model, test_loader, device, class_names, save_path='results/scratch_predictions.png')
        
    elif args.mode == 'evaluate':
        # 评估已有模型
        if args.model_path is None:
            raise ValueError("评估模式需要提供模型路径 --model_path")
        
        print(f"评估模型: {args.model_path}")
        model = create_alexnet_model(num_classes=num_classes)
        model = load_model(model, args.model_path, device)
        model = model.to(device)
        
        # 评估模型
        criterion = nn.CrossEntropyLoss()
        acc, loss, preds, labels = evaluate_model(model, test_loader, device, criterion)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(labels, preds, class_names, save_path='results/evaluation_confusion_matrix.png')
        
        # 生成分类报告
        generate_classification_report(labels, preds, class_names, save_path='results/evaluation_classification_report.txt')
        
        # 可视化模型预测结果
        visualize_model_predictions(model, test_loader, device, class_names, save_path='results/evaluation_predictions.png')
        
        # 可视化特征图
        visualize_feature_maps(model, test_loader, device, save_path='results/feature_maps.png')
        
    elif args.mode == 'compare':
        # 比较微调和从头训练的模型
        if args.model_path is None or args.model_scratch_path is None:
            raise ValueError("比较模式需要提供两个模型路径: --model_path 和 --model_scratch_path")
        
        print(f"比较模型: {args.model_path} 和 {args.model_scratch_path}")
        
        # 加载微调模型
        finetune_model = create_alexnet_model(num_classes=num_classes)
        finetune_model = load_model(finetune_model, args.model_path, device)
        finetune_model = finetune_model.to(device)
        
        # 加载从头训练模型
        scratch_model = create_alexnet_model(num_classes=num_classes)
        scratch_model = load_model(scratch_model, args.model_scratch_path, device)
        scratch_model = scratch_model.to(device)
        
        # 评估微调模型
        criterion = nn.CrossEntropyLoss()
        finetune_acc, finetune_loss, finetune_preds, finetune_labels = evaluate_model(
            finetune_model, test_loader, device, criterion
        )
        
        # 评估从头训练模型
        scratch_acc, scratch_loss, scratch_preds, scratch_labels = evaluate_model(
            scratch_model, test_loader, device, criterion
        )
        
        print(f"微调模型准确率: {finetune_acc:.4f}, 从头训练模型准确率: {scratch_acc:.4f}")
        print(f"准确率提升: {finetune_acc - scratch_acc:.4f}")
        
        # 加载训练历史进行比较
        try:
            with open('results/finetune_history.json', 'r') as f:
                finetune_history = json.load(f)
            
            with open('results/scratch_history.json', 'r') as f:
                scratch_history = json.load(f)
            
            # 比较学习曲线
            compare_learning_curves(
                finetune_history, scratch_history,
                labels=['微调', '从头训练'],
                save_path='results/compare_learning_curves.png'
            )
        except:
            print("无法加载训练历史进行比较")

if __name__ == '__main__':
    main() 