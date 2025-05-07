import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, train_loader, val_loader, 
                optimizer, criterion, num_epochs, 
                device, model_save_path, log_dir,
                model_name):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        criterion: 损失函数
        num_epochs: 训练轮数
        device: 训练设备
        model_save_path: 模型保存路径
        log_dir: TensorBoard日志目录
        model_name: 模型名称
        
    Returns:
        model: 训练后的模型
    """
    since = time.time()
    
    # 创建TensorBoard日志目录
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 创建模型保存目录
    os.makedirs(model_save_path, exist_ok=True)
    
    # 初始化最佳模型权重和精度
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                dataloader = train_loader
            else:
                model.eval()   # 设置模型为验证模式
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # 遍历数据
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 如果是训练阶段则反向传播+优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # 计算epoch损失和准确率
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 记录到TensorBoard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            
            # 如果是验证阶段，且当前模型更好，则保存模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                # 保存当前最佳模型
                torch.save(model.state_dict(), os.path.join(model_save_path, f'{model_name}_best.pth'))
        
        # 每个epoch结束后保存中间模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, os.path.join(model_save_path, f'{model_name}_epoch_{epoch}.pth'))
        
        print()
    
    # 训练结束
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    # 关闭TensorBoard写入器
    writer.close()
    
    return model


def validate_model(model, val_loader, criterion, device):
    """
    验证模型性能
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 验证设备
        
    Returns:
        loss: 损失
        accuracy: 准确率
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    # 计算整体损失和准确率
    loss = running_loss / len(val_loader.dataset)
    accuracy = running_corrects.double() / len(val_loader.dataset)
    
    print(f'Validation Loss: {loss:.4f} Acc: {accuracy:.4f}')
    
    return loss, accuracy 