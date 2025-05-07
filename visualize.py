import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision

def show_batch(dataloader, class_names=None, save_path=None):
    """
    显示一个批次的图像
    
    Args:
        dataloader: 数据加载器
        class_names: 类别名称
        save_path: 保存路径
    """
    # 获取一批数据
    inputs, labels = next(iter(dataloader))
    
    # 计算图像网格
    grid_size = int(np.ceil(np.sqrt(inputs.size(0))))
    
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = inputs * std + mean
    
    # 创建图像网格
    plt.figure(figsize=(15, 15))
    for i in range(min(25, inputs.size(0))):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        if class_names:
            plt.title(class_names[labels[i]])
        plt.axis("off")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def log_model_graph(model, dataloader, log_dir):
    """
    记录模型图到TensorBoard
    
    Args:
        model: 模型
        dataloader: 数据加载器
        log_dir: 日志目录
    """
    writer = SummaryWriter(log_dir)
    
    # 获取一批数据
    inputs, _ = next(iter(dataloader))
    
    # 添加模型图
    writer.add_graph(model, inputs)
    
    writer.close()

def visualize_model_predictions(model, dataloader, class_names, device, num_images=6, save_path=None):
    """
    可视化模型预测
    
    Args:
        model: 模型
        dataloader: 数据加载器
        class_names: 类别名称
        device: 设备
        num_images: 显示图像数量
        save_path: 保存路径
    """
    model.eval()
    
    # 获取一批数据
    inputs, labels = next(iter(dataloader))
    
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = inputs * std + mean
    
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
    
    # 绘制图像和预测
    plt.figure(figsize=(15, 10))
    for i in range(min(num_images, inputs.size(0))):
        plt.subplot(2, num_images//2, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        
        color = 'green' if preds[i] == labels[i] else 'red'
        title = f'Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}'
        
        plt.title(title, color=color)
        plt.axis("off")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    绘制训练历史
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_path: 保存路径
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def log_images_to_tensorboard(writer, dataloader, tag, global_step=0):
    """
    记录图像到TensorBoard
    
    Args:
        writer: TensorBoard写入器
        dataloader: 数据加载器
        tag: 标签
        global_step: 全局步数
    """
    # 获取一批数据
    inputs, labels = next(iter(dataloader))
    
    # 创建网格图像
    grid = torchvision.utils.make_grid(inputs)
    
    # 添加图像
    writer.add_image(tag, grid, global_step)
    
    return writer 