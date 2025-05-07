import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
import glob

class Caltech101Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, test_ratio=0.2, random_state=42):
        """
        Caltech-101 数据集加载器
        
        参数:
            root_dir (string): 数据集根目录
            split (string): 'train' 或 'test'
            transform (callable, optional): 样本转换
            test_ratio (float): 测试集比例
            random_state (int): 随机种子
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 收集所有图像文件路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_path in glob.glob(os.path.join(class_dir, '*.jpg')):
                self.samples.append((img_path, self.class_to_idx[class_name]))
        
        # 划分训练集和测试集
        train_samples, test_samples = train_test_split(
            self.samples, test_size=test_ratio, random_state=random_state, stratify=[s[1] for s in self.samples]
        )
        
        self.samples = train_samples if split == 'train' else test_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms():
    """
    获取数据预处理转换
    
    返回:
        transform_train: 训练集转换
        transform_test: 测试集转换
    """
    # AlexNet 输入尺寸为 224x224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    return transform_train, transform_test

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    创建数据加载器
    
    参数:
        data_dir (string): 数据集目录
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        
    返回:
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        num_classes: 类别数量
    """
    transform_train, transform_test = get_data_transforms()
    
    train_dataset = Caltech101Dataset(data_dir, 'train', transform_train)
    test_dataset = Caltech101Dataset(data_dir, 'test', transform_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, len(train_dataset.classes) 