import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

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

def plot_confusion_matrix(true_labels, predictions, class_names=None, save_path=None, 
                          normalize=True, figsize=(16, 14), dpi=100, max_classes=30):
    """
    绘制混淆矩阵
    
    Args:
        true_labels: 真实标签
        predictions: 预测标签
        class_names: 类别名称
        save_path: 保存路径
        normalize: 是否归一化
        figsize: 图像大小
        dpi: 图像分辨率
        max_classes: 最大显示类别数量
    """
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    
    # 如果类别数量太多，选择表现最差的max_classes个类别
    if class_names and len(class_names) > max_classes:
        print(f"类别太多，只显示表现最差的{max_classes}个类别的混淆矩阵")
        
        # 计算每个类别的准确率
        class_acc = []
        for i in range(len(class_names)):
            # 计算第i类的准确率
            idx = true_labels == i
            if np.sum(idx) > 0:  # 避免除以零
                acc = np.sum(np.array(predictions)[idx] == i) / np.sum(idx)
                class_acc.append((i, acc))
        
        # 按准确率排序，选择最差的max_classes个类别
        class_acc.sort(key=lambda x: x[1])
        worst_classes = [x[0] for x in class_acc[:max_classes]]
        
        # 提取这些类别的混淆矩阵
        cm_subset = cm[worst_classes, :][:, worst_classes]
        cm = cm_subset
        if class_names:
            class_names = [class_names[i] for i in worst_classes]
    
    # 归一化混淆矩阵
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'
    
    # 创建图形
    plt.figure(figsize=figsize, dpi=dpi)
    
    # 创建热图
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    # 调整标签
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    
    # 旋转标签以便阅读
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45)
    
    # 调整布局，确保所有标签都可见
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
        # 同时保存完整混淆矩阵为CSV文件
        if class_names and len(class_names) > max_classes:
            # 创建完整混淆矩阵的DataFrame
            full_cm = confusion_matrix(true_labels, predictions)
            all_class_names = [class_names[i] if i < len(class_names) else f"Class {i}" 
                             for i in range(full_cm.shape[0])]
            cm_df = pd.DataFrame(full_cm, index=all_class_names, columns=all_class_names)
            
            # 保存CSV
            csv_path = os.path.splitext(save_path)[0] + '_full.csv'
            cm_df.to_csv(csv_path)
            print(f"完整混淆矩阵已保存至: {csv_path}")
    
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
    report = classification_report(true_labels, predictions, target_names=class_names, digits=4)
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

def plot_top_misclassifications(true_labels, predictions, class_names, top_n=10, save_path=None):
    """
    绘制最常见的错误分类
    
    Args:
        true_labels: 真实标签
        predictions: 预测标签
        class_names: 类别名称
        top_n: 显示前N个最常见的错误
        save_path: 保存路径
    """
    # 获取不正确的预测
    mask = true_labels != predictions
    misclassified_true = true_labels[mask]
    misclassified_pred = predictions[mask]
    
    # 创建错误对的列表
    error_pairs = list(zip(misclassified_true, misclassified_pred))
    
    # 计算每种错误的频率
    error_counts = {}
    for true_label, pred_label in error_pairs:
        key = (true_label, pred_label)
        if key in error_counts:
            error_counts[key] += 1
        else:
            error_counts[key] = 1
    
    # 排序并获取前N个最常见的错误
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    top_errors = sorted_errors[:top_n]
    
    # 准备绘图数据
    true_classes = [class_names[true] for (true, _), _ in top_errors]
    pred_classes = [class_names[pred] for (_, pred), _ in top_errors]
    counts = [count for _, count in top_errors]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建水平条形图
    y_pos = np.arange(len(true_classes))
    plt.barh(y_pos, counts, align='center', color='crimson', alpha=0.7)
    
    # 添加标签
    for i, (true, pred, count) in enumerate(zip(true_classes, pred_classes, counts)):
        plt.text(count + 0.5, i, f"{count} ({true} → {pred})", va='center')
    
    plt.xlabel('错误次数')
    plt.title('最常见的错误分类')
    plt.yticks([])  # 隐藏y轴刻度，因为我们在条形上有标签
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show() 