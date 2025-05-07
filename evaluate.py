import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_model(model, val_loader, criterion, device, class_names=None):
    """
    评估模型性能
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 评估设备
        class_names: 类别名称列表
        
    Returns:
        loss: 损失
        accuracy: 准确率
        all_preds: 所有预测
        all_labels: 所有标签
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # 存储预测和标签用于后续分析
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算整体损失和准确率
    loss = running_loss / len(val_loader.dataset)
    accuracy = running_corrects.double() / len(val_loader.dataset)
    
    print(f'Evaluation Loss: {loss:.4f} Acc: {accuracy:.4f}')
    
    return loss, accuracy, np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(true_labels, predictions, class_names=None, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        true_labels: 真实标签
        predictions: 预测标签
        class_names: 类别名称
        save_path: 保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    
    # 如果类别数量太多，只显示前20个
    if class_names and len(class_names) > 20:
        print("类别太多，只显示前20个类别的混淆矩阵")
        cm = cm[:20, :20]
        class_names = class_names[:20]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def generate_classification_report(true_labels, predictions, class_names=None, save_path=None):
    """
    生成分类报告
    
    Args:
        true_labels: 真实标签
        predictions: 预测标签
        class_names: 类别名称
        save_path: 保存路径
    """
    # 生成分类报告
    report = classification_report(true_labels, predictions, target_names=class_names)
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)

def compare_models(model_results, save_path=None):
    """
    比较不同模型的性能
    
    Args:
        model_results: 模型结果字典，格式为 {model_name: (accuracy, loss)}
        save_path: 保存路径
    """
    # 提取模型名称、准确率和损失
    model_names = list(model_results.keys())
    accuracies = [result[0] for result in model_results.values()]
    losses = [result[1] for result in model_results.values()]
    
    # 绘制准确率对比
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(model_names, accuracies, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    
    # 绘制损失对比
    plt.subplot(1, 2, 2)
    plt.bar(model_names, losses, color='salmon')
    plt.xlabel('Model')
    plt.ylabel('Loss')
    plt.title('Model Loss Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show() 