import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image

def plot_learning_curves(history, save_path=None):
    """
    绘制学习曲线
    
    参数:
        history: 训练历史记录
        save_path: 保存路径，如果为None则只显示不保存
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()

def visualize_model_predictions(model, dataloader, device, class_names, num_images=6, save_path=None):
    """
    可视化模型预测结果
    
    参数:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        class_names: 类别名称
        num_images: 可视化的图像数量
        save_path: 保存路径，如果为None则只显示不保存
    """
    was_training = model.training
    model.eval()
    
    images_so_far = 0
    fig = plt.figure(figsize=(15, 12))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}\ntrue: {class_names[labels[j]]}')
                
                # 将张量转为图像显示
                inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                
                plt.imshow(inp)
                
                if images_so_far == num_images:
                    if save_path:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        plt.savefig(save_path)
                    model.train(mode=was_training)
                    return
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        model.train(mode=was_training)

def compare_learning_curves(history1, history2, labels=['Finetune', 'Scratch'], save_path=None):
    """
    比较两个模型的学习曲线
    
    参数:
        history1: 第一个模型的训练历史
        history2: 第二个模型的训练历史
        labels: 两个模型的标签
        save_path: 保存路径，如果为None则只显示不保存
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history1['val_loss'], label=labels[0])
    plt.plot(history2['val_loss'], label=labels[1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history1['val_acc'], label=labels[0])
    plt.plot(history2['val_acc'], label=labels[1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()

def visualize_feature_maps(model, dataloader, device, layer_name='features.10', num_images=1, save_path=None):
    """
    可视化网络特征图
    
    参数:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        layer_name: 要可视化的层名称
        num_images: 可视化的图像数量
        save_path: 保存路径，如果为None则只显示不保存
    """
    was_training = model.training
    model.eval()
    
    # 存储特征图的钩子
    features_blobs = []
    
    def hook_feature(module, input, output):
        features_blobs.append(output.detach().cpu())
    
    # 注册钩子
    for name, m in model.named_modules():
        if name == layer_name:
            m.register_forward_hook(hook_feature)
    
    # 获取一批数据
    dataiter = iter(dataloader)
    inputs, labels = next(dataiter)
    inputs = inputs.to(device)
    
    # 前向传播
    with torch.no_grad():
        _ = model(inputs[:num_images])
    
    # 可视化特征图
    if len(features_blobs) > 0:
        features = features_blobs[0]
        
        for i in range(num_images):
            plt.figure(figsize=(20, 10))
            feat_map = features[i]
            
            grid_size = int(np.ceil(np.sqrt(feat_map.size(0))))
            grid_img = vutils.make_grid(feat_map.unsqueeze(1), normalize=True, nrow=grid_size)
            
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.title(f'Feature Maps - {layer_name}')
            plt.axis('off')
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(f"{save_path.split('.')[0]}_{i}.{save_path.split('.')[1]}")
            
            plt.show()
    
    model.train(mode=was_training) 