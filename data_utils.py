import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

def get_data_transforms():
    """
    获取数据转换
    """
    # 训练集转换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集转换
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_caltech101_data(batch_size=32, num_workers=4, data_dir='data'):
    """
    加载Caltech-101数据集
    
    参数:
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        data_dir: 数据目录
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        class_names: 类别名称列表
    """
    # 获取数据转换
    train_transform, val_transform = get_data_transforms()
    
    # 下载并加载数据集
    dataset = datasets.Caltech101(
        root=data_dir,
        download=True,
        transform=train_transform
    )
    
    # 获取类别名称
    class_names = dataset.categories
    
    # 计算数据集大小
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # 随机分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 为验证集和测试集设置不同的转换
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names

def visualize_batch(images, labels, class_names, num_images=8):
    """
    可视化一个批次的数据
    
    参数:
        images: 图像张量
        labels: 标签张量
        class_names: 类别名称列表
        num_images: 要显示的图像数量
    """
    # 将图像转换为numpy数组
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images.transpose(0, 2, 3, 1) + mean
    images = np.clip(images, 0, 1)
    
    # 创建图像网格
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        axes[i].imshow(images[i])
        axes[i].set_title(class_names[labels[i]])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_class_distribution(dataset):
    """
    获取数据集的类别分布
    
    参数:
        dataset: 数据集
    
    返回:
        class_counts: 类别计数字典
    """
    class_counts = {}
    for _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    return class_counts 