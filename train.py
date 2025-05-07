import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, 
                num_epochs=25, is_inception=False, model_save_path='models',
                tensorboard_dir='runs', model_name='model'):
    """
    训练模型函数
    
    参数:
        model: 待训练的模型
        dataloaders: 数据加载器字典 {'train': train_loader, 'val': val_loader}
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 训练设备
        num_epochs: 训练轮数
        is_inception: 是否为inception模型
        model_save_path: 模型保存路径
        tensorboard_dir: TensorBoard日志路径
        model_name: 模型名称
        
    返回:
        model: 训练后的模型
        history: 训练历史
    """
    os.makedirs(model_save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(tensorboard_dir, model_name + '_' + time.strftime("%Y%m%d-%H%M%S")))
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)
        
        # 每个 epoch 都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式
                
            running_loss = 0.0
            running_corrects = 0
            
            # 遍历数据
            pbar = tqdm(dataloaders[phase], desc=f'{phase} batch')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # 反向传播 + 优化（仅在训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                pbar.set_postfix({'loss': loss.item()})
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 记录损失和准确率
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                history['train_loss'].append(epoch_loss)
            else:
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # 深度复制模型（如果是最好的结果）
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'acc': epoch_acc,
                }, os.path.join(model_save_path, f'{model_name}_best.pth'))
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'acc': epoch_acc,
    }, os.path.join(model_save_path, f'{model_name}_final.pth'))
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history

def fine_tune_alexnet(model, dataloaders, device, 
                      learning_rate=0.001, num_epochs=25, feature_extract=False,
                      model_save_path='models', tensorboard_dir='runs'):
    """
    微调AlexNet
    
    参数:
        model: 预训练的AlexNet模型
        dataloaders: 数据加载器字典
        device: 训练设备
        learning_rate: 基础学习率
        num_epochs: 训练轮数
        feature_extract: 是否只训练最后一层
        model_save_path: 模型保存路径
        tensorboard_dir: TensorBoard日志路径
        
    返回:
        model: 微调后的模型
        history: 训练历史
    """
    from model import get_fine_tuning_parameters
    
    params_to_update = get_fine_tuning_parameters(model, feature_extract)
    
    # 为不同参数组设置不同的学习率
    if isinstance(params_to_update, list) and len(params_to_update) > 1:
        optimizer = optim.SGD([
            {'params': params_to_update[0]['params'], 'lr': learning_rate * params_to_update[0]['lr_mult']},
            {'params': params_to_update[1]['params'], 'lr': learning_rate * params_to_update[1]['lr_mult']}
        ], momentum=0.9)
    else:
        optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, device,
        num_epochs=num_epochs, model_save_path=model_save_path,
        tensorboard_dir=tensorboard_dir, model_name='alexnet_finetune' if feature_extract else 'alexnet_full_finetune'
    )
    
    return model, history

def train_alexnet_from_scratch(model, dataloaders, device, 
                              learning_rate=0.01, num_epochs=25,
                              model_save_path='models', tensorboard_dir='runs'):
    """
    从头训练AlexNet
    
    参数:
        model: 未预训练的AlexNet模型
        dataloaders: 数据加载器字典
        device: 训练设备
        learning_rate: 学习率
        num_epochs: 训练轮数
        model_save_path: 模型保存路径
        tensorboard_dir: TensorBoard日志路径
        
    返回:
        model: 训练后的模型
        history: 训练历史
    """
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, device,
        num_epochs=num_epochs, model_save_path=model_save_path,
        tensorboard_dir=tensorboard_dir, model_name='alexnet_scratch'
    )
    
    return model, history 