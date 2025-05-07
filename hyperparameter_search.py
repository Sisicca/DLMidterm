import os
import json
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_utils import get_data_loaders
from model import create_alexnet_model
from train import fine_tune_alexnet, train_alexnet_from_scratch
from evaluate import evaluate_model

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def hyperparameter_search():
    """
    超参数搜索函数
    """
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs('models/hyperparameter_search', exist_ok=True)
    os.makedirs('results/hyperparameter_search', exist_ok=True)
    
    # 加载数据
    train_loader, test_loader, num_classes = get_data_loaders(
        'data/caltech101', batch_size=32, num_workers=4
    )
    
    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }
    
    # 定义超参数搜索空间
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    epochs_list = [20, 30, 40]
    feature_extract_options = [True, False]  # True: 只训练最后一层, False: 微调所有层
    
    # 记录最佳超参数和性能
    best_accuracy = 0.0
    best_params = {}
    results = []
    
    # 超参数搜索
    for lr in learning_rates:
        for num_epochs in epochs_list:
            for feature_extract in feature_extract_options:
                print(f"\n超参数组合: lr={lr}, epochs={num_epochs}, feature_extract={feature_extract}")
                
                # 创建模型
                model = create_alexnet_model(num_classes=num_classes, pretrained=True)
                model = model.to(device)
                
                # 微调模型
                model_name = f"alexnet_lr{lr}_ep{num_epochs}_fe{int(feature_extract)}"
                model, history = fine_tune_alexnet(
                    model, dataloaders, device,
                    learning_rate=lr,
                    num_epochs=num_epochs,
                    feature_extract=feature_extract,
                    model_save_path='models/hyperparameter_search',
                    tensorboard_dir='runs/hyperparameter_search',
                    
                )
                
                # 评估模型
                criterion = nn.CrossEntropyLoss()
                accuracy, loss, _, _ = evaluate_model(model, test_loader, device, criterion)
                
                # 记录结果
                result = {
                    'learning_rate': lr,
                    'num_epochs': num_epochs,
                    'feature_extract': feature_extract,
                    'accuracy': accuracy,
                    'loss': loss,
                    'model_name': model_name
                }
                results.append(result)
                
                # 更新最佳超参数
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'learning_rate': lr,
                        'num_epochs': num_epochs,
                        'feature_extract': feature_extract,
                        'model_name': model_name
                    }
    
    # 保存结果
    with open('results/hyperparameter_search/results.json', 'w') as f:
        json.dump({
            'results': results,
            'best_params': best_params,
            'best_accuracy': best_accuracy
        }, f, indent=4)
    
    # 可视化超参数搜索结果
    visualize_hyperparameter_search(results)
    
    print(f"\n超参数搜索完成")
    print(f"最佳超参数: {best_params}")
    print(f"最佳准确率: {best_accuracy:.4f}")
    
    return best_params

def visualize_hyperparameter_search(results):
    """
    可视化超参数搜索结果
    
    参数:
        results: 超参数搜索结果列表
    """
    # 提取数据
    lrs = sorted(list(set([r['learning_rate'] for r in results])))
    epochs = sorted(list(set([r['num_epochs'] for r in results])))
    
    # 创建图表
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制学习率对准确率的影响
    lr_acc_fe_true = [r['accuracy'] for r in results if r['feature_extract'] == True]
    lr_acc_fe_false = [r['accuracy'] for r in results if r['feature_extract'] == False]
    
    lr_indices = np.arange(len(lrs))
    width = 0.35
    
    axs[0].bar(lr_indices - width/2, lr_acc_fe_true[:len(lrs)], width, label='只训练最后一层')
    axs[0].bar(lr_indices + width/2, lr_acc_fe_false[:len(lrs)], width, label='微调所有层')
    
    axs[0].set_xlabel('学习率')
    axs[0].set_ylabel('准确率')
    axs[0].set_title('学习率对准确率的影响')
    axs[0].set_xticks(lr_indices)
    axs[0].set_xticklabels([str(lr) for lr in lrs])
    axs[0].legend()
    
    # 绘制训练轮数对准确率的影响
    epoch_acc_fe_true = []
    epoch_acc_fe_false = []
    
    for epoch in epochs:
        epoch_acc_fe_true.append(np.mean([r['accuracy'] for r in results if r['num_epochs'] == epoch and r['feature_extract'] == True]))
        epoch_acc_fe_false.append(np.mean([r['accuracy'] for r in results if r['num_epochs'] == epoch and r['feature_extract'] == False]))
    
    epoch_indices = np.arange(len(epochs))
    
    axs[1].bar(epoch_indices - width/2, epoch_acc_fe_true, width, label='只训练最后一层')
    axs[1].bar(epoch_indices + width/2, epoch_acc_fe_false, width, label='微调所有层')
    
    axs[1].set_xlabel('训练轮数')
    axs[1].set_ylabel('准确率')
    axs[1].set_title('训练轮数对准确率的影响')
    axs[1].set_xticks(epoch_indices)
    axs[1].set_xticklabels([str(epoch) for epoch in epochs])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig('results/hyperparameter_search/visualization.png')
    plt.close()

if __name__ == '__main__':
    hyperparameter_search() 