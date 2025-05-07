import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, dataloader, device, criterion=None):
    """
    评估模型性能
    
    参数:
        model: 待评估的模型
        dataloader: 测试数据加载器
        device: 评估设备
        criterion: 损失函数，如果为None则只计算准确率
        
    返回:
        accuracy: 准确率
        loss: 损失（如果criterion不为None）
        preds: 预测结果
        labels: 真实标签
    """
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = running_corrects.double() / len(dataloader.dataset)
    
    if criterion is not None:
        loss = running_loss / len(dataloader.dataset)
        print(f'Test Loss: {loss:.4f} Accuracy: {accuracy:.4f}')
        return accuracy.item(), loss, all_preds, all_labels
    else:
        print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy.item(), None, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称
        save_path: 保存路径，如果为None则只显示不保存
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()

def generate_classification_report(y_true, y_pred, classes, save_path=None):
    """
    生成分类报告
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称
        save_path: 保存路径，如果为None则只显示不保存
    """
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)

def load_model(model, model_path, device):
    """
    加载模型
    
    参数:
        model: 模型
        model_path: 模型路径
        device: 设备
        
    返回:
        model: 加载参数后的模型
    """
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model 