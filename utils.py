import torch
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR, ReduceLROnPlateau

def create_optimizer(optimizer_name, model_params, lr, weight_decay=0.0):
    """
    优化器工厂函数，根据名称创建相应的优化器
    
    Args:
        optimizer_name: 优化器名称
        model_params: 模型参数
        lr: 学习率
        weight_decay: 权重衰减
        
    Returns:
        optimizer: 优化器实例
    """
    optimizers = {
        'sgd': lambda: optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay),
        'adam': lambda: optim.Adam(model_params, lr=lr, weight_decay=weight_decay),
        'adamw': lambda: optim.AdamW(model_params, lr=lr, weight_decay=weight_decay),
        'rmsprop': lambda: optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay),
        'adadelta': lambda: optim.Adadelta(model_params, lr=lr, weight_decay=weight_decay),
        'adagrad': lambda: optim.Adagrad(model_params, lr=lr, weight_decay=weight_decay)
    }
    
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"不支持的优化器: {optimizer_name}. 支持的优化器: {list(optimizers.keys())}")
    
    return optimizers[optimizer_name.lower()]()

def create_scheduler(scheduler_name, optimizer, epochs, steps_per_epoch, warmup_steps=0, 
                    min_lr=1e-6, patience=5):
    """
    学习率调度器工厂函数
    
    Args:
        scheduler_name: 调度器名称
        optimizer: 优化器
        epochs: 总训练轮数
        steps_per_epoch: 每个epoch的步数
        warmup_steps: 预热步数
        min_lr: 最小学习率
        patience: 学习率下降前等待的轮数
        
    Returns:
        scheduler: 学习率调度器
    """
    total_steps = epochs * steps_per_epoch
    
    # 线性预热函数
    def warmup_function(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    
    # 线性衰减函数
    def linear_decay_with_warmup(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    
    # 余弦衰减函数
    def cosine_decay_with_warmup(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    schedulers = {
        'constant': lambda: LambdaLR(optimizer, lambda _: 1.0),
        'warmup': lambda: LambdaLR(optimizer, warmup_function),
        'linear_decay': lambda: LambdaLR(optimizer, lambda step: max(min_lr, 1.0 - step / total_steps)),
        'linear_warmup_decay': lambda: LambdaLR(optimizer, linear_decay_with_warmup),
        'cosine': lambda: CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr),
        'cosine_warmup': lambda: LambdaLR(optimizer, cosine_decay_with_warmup),
        'step': lambda: StepLR(optimizer, step_size=steps_per_epoch * (epochs // 3), gamma=0.1),
        'reduce_on_plateau': lambda: ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                     patience=patience, min_lr=min_lr)
    }
    
    if scheduler_name.lower() not in schedulers:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}. 支持的调度器: {list(schedulers.keys())}")
    
    return schedulers[scheduler_name.lower()]()

class Logger:
    """简单的日志记录器，支持不同级别的日志信息"""
    
    LEVELS = {'debug': 0, 'info': 1, 'warning': 2, 'error': 3}
    
    def __init__(self, log_level='info', log_file=None):
        self.log_level = self.LEVELS.get(log_level.lower(), 1)
        self.log_file = log_file
        if log_file:
            with open(log_file, 'w') as f:
                f.write(f"开始记录日志...\n")
    
    def log(self, message, level='info'):
        level_num = self.LEVELS.get(level.lower(), 1)
        if level_num >= self.log_level:
            log_message = f"[{level.upper()}] {message}"
            print(log_message)
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(f"{log_message}\n")
    
    def debug(self, message):
        self.log(message, 'debug')
    
    def info(self, message):
        self.log(message, 'info')
    
    def warning(self, message):
        self.log(message, 'warning')
    
    def error(self, message):
        self.log(message, 'error')

class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self, name):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}" 