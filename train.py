import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import AverageMeter, Logger

def train_model(model, train_loader, val_loader, 
                optimizer, criterion, num_epochs, 
                device, model_save_path, log_dir,
                model_name, scheduler=None, 
                log_frequency=10, grad_clip=None,
                save_frequency=1, mixed_precision=False,
                early_stopping=None):
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
        scheduler: 学习率调度器
        log_frequency: 日志打印频率（每N个批次）
        grad_clip: 梯度裁剪值，None表示不裁剪
        save_frequency: 模型保存频率（每N个epoch）
        mixed_precision: 是否使用混合精度训练
        early_stopping: 早停轮数，None表示不使用早停
        
    Returns:
        model: 训练后的模型
    """
    since = time.time()
    
    # 创建TensorBoard日志目录
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 创建模型保存目录
    os.makedirs(model_save_path, exist_ok=True)
    
    # 创建日志记录器
    logger = Logger(log_level='info', log_file=os.path.join(log_dir, 'training.log'))
    logger.info(f"开始训练: {model_name}")
    logger.info(f"训练设备: {device}")
    logger.info(f"训练轮数: {num_epochs}")
    logger.info(f"批次大小: {train_loader.batch_size}")
    logger.info(f"优化器: {optimizer.__class__.__name__}")
    if scheduler:
        logger.info(f"学习率调度器: {scheduler.__class__.__name__}")
    
    # 初始化最佳模型权重和精度
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_epoch = 0
    
    # 初始化早停计数器
    early_stop_counter = 0
    
    # 如果使用混合精度训练，初始化scaler
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("使用混合精度训练")
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)
        
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                dataloader = train_loader
            else:
                model.eval()   # 设置模型为验证模式
                dataloader = val_loader
            
            # 初始化指标跟踪器
            losses = AverageMeter('Loss')
            accuracies = AverageMeter('Accuracy')
            
            # 遍历数据
            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc=phase)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    # 如果使用混合精度训练
                    if mixed_precision and phase == 'train':
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # 如果是训练阶段则反向传播+优化
                    if phase == 'train':
                        if mixed_precision:
                            scaler.scale(loss).backward()
                            if grad_clip is not None:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            if grad_clip is not None:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                            optimizer.step()
                
                # 统计
                batch_loss = loss.item()
                batch_corrects = torch.sum(preds == labels.data).double() / inputs.size(0)
                batch_size = inputs.size(0)
                
                losses.update(batch_loss, batch_size)
                accuracies.update(batch_corrects.item(), batch_size)
                
                # 定期打印训练信息
                if phase == 'train' and (batch_idx + 1) % log_frequency == 0:
                    # 获取当前学习率
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    logger.debug(
                        f'Epoch: [{epoch+1}/{num_epochs}][{batch_idx+1}/{len(dataloader)}] '
                        f'Loss: {losses.val:.4f} ({losses.avg:.4f}) '
                        f'Acc: {accuracies.val:.4f} ({accuracies.avg:.4f}) '
                        f'LR: {current_lr:.6f}'
                    )
                    
                    # 记录到TensorBoard
                    global_step = epoch * len(dataloader) + batch_idx
                    writer.add_scalar('Batch/Loss', losses.val, global_step)
                    writer.add_scalar('Batch/Accuracy', accuracies.val, global_step)
                    writer.add_scalar('Batch/LR', current_lr, global_step)
                    
                    # 计算梯度范数
                    if phase == 'train' and not mixed_precision:
                        grad_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                grad_norm += p.grad.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5
                        writer.add_scalar('Batch/GradNorm', grad_norm, global_step)
            
            # 计算epoch平均损失和准确率
            epoch_loss = losses.avg
            epoch_acc = accuracies.avg
            
            # 输出epoch结果
            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 记录到TensorBoard
            writer.add_scalar(f'Epoch/{phase}_Loss', epoch_loss, epoch)
            writer.add_scalar(f'Epoch/{phase}_Accuracy', epoch_acc, epoch)
            
            # 如果是验证阶段，且当前模型更好，则保存模型
            if phase == 'val' and epoch_acc > best_acc:
                logger.info(f'验证准确率提高: {best_acc:.4f} -> {epoch_acc:.4f}')
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                best_epoch = epoch
                # 保存当前最佳模型
                torch.save(model.state_dict(), os.path.join(model_save_path, f'{model_name}_best.pth'))
                # 重置早停计数器
                early_stop_counter = 0
            elif phase == 'val' and early_stopping is not None:
                # 如果验证性能没有提高，增加早停计数器
                early_stop_counter += 1
                if early_stop_counter >= early_stopping:
                    logger.info(f'触发早停: {early_stopping}轮验证准确率未提高')
                    logger.info(f'最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch+1})')
                    # 提前结束训练
                    model.load_state_dict(best_model_wts)
                    return model
        
        # 更新学习率调度器
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()
                
            # 记录新的学习率
            new_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Epoch/LR', new_lr, epoch)
            logger.debug(f'学习率更新为: {new_lr:.6f}')
        
        # 每N个epoch保存模型
        if (epoch + 1) % save_frequency == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(model_save_path, f'{model_name}_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': epoch_loss,
                'acc': epoch_acc,
                'best_acc': best_acc,
            }, checkpoint_path)
            logger.debug(f'模型保存至: {checkpoint_path}')
        
        # 计算并记录epoch耗时
        epoch_time = time.time() - epoch_start_time
        logger.info(f'Epoch {epoch+1} 耗时: {epoch_time:.2f}s')
        logger.info('')
    
    # 训练结束
    time_elapsed = time.time() - since
    logger.info(f'训练完成，耗时: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    logger.info(f'最佳验证准确率: {best_acc:.4f} (Epoch {best_epoch+1})')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    # 关闭TensorBoard写入器
    writer.close()
    
    return model


def validate_model(model, val_loader, criterion, device, logger=None):
    """
    验证模型性能
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 验证设备
        logger: 日志记录器
        
    Returns:
        loss: 损失
        accuracy: 准确率
    """
    model.eval()
    losses = AverageMeter('Loss')
    accuracies = AverageMeter('Accuracy')
    
    if logger is None:
        logger = Logger(log_level='info')
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            batch_size = inputs.size(0)
            batch_loss = loss.item()
            batch_acc = torch.sum(preds == labels.data).double() / batch_size
            
            losses.update(batch_loss, batch_size)
            accuracies.update(batch_acc.item(), batch_size)
    
    # 计算整体损失和准确率
    val_loss = losses.avg
    val_acc = accuracies.avg
    
    logger.info(f'验证集结果: Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    return val_loss, val_acc 