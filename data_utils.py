import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_transforms():
    """
    返回训练和测试数据的转换
    """
    # AlexNet的预处理要求
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def load_caltech101_data(data_dir='caltech-101/101_ObjectCategories', batch_size=64, train_ratio=0.8):
    """
    加载Caltech-101数据集
    
    Args:
        data_dir: 数据集目录
        batch_size: 批处理大小
        train_ratio: 训练集比例
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        class_to_idx: 类别到索引的映射
    """
    train_transform, test_transform = get_data_transforms()
    
    # 加载全部数据
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    # 按照标准划分训练集和测试集
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    # 随机分割数据集
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 更新验证集的转换
    val_dataset.dataset.transform = test_transform
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.class_to_idx

def get_num_classes(data_dir='caltech-101/101_ObjectCategories'):
    """获取数据集的类别数量"""
    return len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]) 