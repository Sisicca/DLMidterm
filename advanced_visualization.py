import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
import cv2
from tqdm import tqdm

class GradCAM:
    """
    使用Grad-CAM算法可视化模型决策的热力图
    
    参考: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/abs/1610.02391
    """
    
    def __init__(self, model, target_layer):
        """
        初始化Grad-CAM
        
        Args:
            model: 要解释的模型
            target_layer: 目标卷积层
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.register_hooks()
        
    def register_hooks(self):
        """注册前向和反向传播的钩子"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 注册钩子
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, target_class=None):
        """
        为输入图像生成Grad-CAM
        
        Args:
            input_image: 输入图像张量 (1, C, H, W)
            target_class: 目标类别索引，None表示使用预测类别
            
        Returns:
            cam: Grad-CAM热力图 (H, W)
            pred_class: 预测的类别索引
            pred_prob: 预测的概率
        """
        # 设置模型为评估模式
        self.model.eval()
        
        # 前向传播
        output = self.model(input_image)
        
        # 如果没有指定目标类别，使用预测类别
        if target_class is None:
            pred_class = torch.argmax(output).item()
            target_class = pred_class
        else:
            pred_class = target_class
        
        # 计算预测概率
        probs = F.softmax(output, dim=1)
        pred_prob = probs[0, pred_class].item()
        
        # 清零所有梯度
        self.model.zero_grad()
        
        # 反向传播
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().cpu().numpy()
        
        # 应用ReLU，只保留正值
        cam = np.maximum(cam, 0)
        
        # 归一化
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        # 调整大小以匹配输入图像
        input_size = input_image.size(-1)
        cam = cv2.resize(cam, (input_size, input_size))
        
        return cam, pred_class, pred_prob
    
    def overlay_cam(self, image, cam, alpha=0.5):
        """
        将CAM热力图覆盖在原始图像上
        
        Args:
            image: 原始图像张量 (C, H, W)
            cam: CAM热力图 (H, W)
            alpha: 混合因子
            
        Returns:
            visualization: 覆盖热力图的图像
        """
        # 将图像从张量转换为numpy数组
        image = image.cpu().numpy().transpose(1, 2, 0)
        
        # 对图像进行反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        # 将CAM转换为热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # 将热力图与原始图像混合
        visualization = (1 - alpha) * image + alpha * heatmap
        visualization = np.clip(visualization, 0, 1)
        
        return visualization


def visualize_features(model, layer_name, input_image, num_features=16, save_path=None):
    """
    可视化网络的特征图
    
    Args:
        model: 模型
        layer_name: 要可视化的层名称
        input_image: 输入图像张量 (1, C, H, W)
        num_features: 要显示的特征数量
        save_path: 保存路径
    """
    # 设置模型为评估模式
    model.eval()
    
    # 初始化特征存储
    features = []
    
    # 定义钩子函数
    def hook_fn(module, input, output):
        features.append(output.detach())
    
    # 查找目标层
    for name, module in model.named_modules():
        if layer_name in name:
            # 注册钩子
            handle = module.register_forward_hook(hook_fn)
            break
    
    # 前向传播
    with torch.no_grad():
        _ = model(input_image)
    
    # 移除钩子
    handle.remove()
    
    # 获取特征
    if not features:
        raise ValueError(f"没有找到名为 {layer_name} 的层")
    
    feature_maps = features[0][0]  # 获取第一个批次的特征
    
    # 计算要显示的行数和列数
    grid_size = int(np.ceil(np.sqrt(num_features)))
    
    # 创建图形
    plt.figure(figsize=(15, 15))
    
    # 显示特征图
    for i in range(min(num_features, feature_maps.size(0))):
        plt.subplot(grid_size, grid_size, i + 1)
        feature_map = feature_maps[i].cpu().numpy()
        plt.imshow(feature_map, cmap='viridis')
        plt.axis('off')
        plt.title(f'Feature {i+1}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def preprocess_image(image_path, size=224):
    """
    预处理图像以供模型使用
    
    Args:
        image_path: 图像路径
        size: 调整大小
        
    Returns:
        原始图像和预处理后的图像张量
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 定义预处理变换
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 预处理图像
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
    
    return image, input_tensor


def visualize_prediction_with_gradcam(model, image_path, target_layer, class_names, device, target_class=None, save_path=None):
    """
    使用Grad-CAM可视化模型预测
    
    Args:
        model: 模型
        image_path: 图像路径
        target_layer: 目标卷积层
        class_names: 类别名称列表
        device: 设备
        target_class: 目标类别索引，None表示使用预测类别
        save_path: 保存路径
    """
    # 预处理图像
    original_image, input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # 初始化Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # 生成CAM
    cam, pred_class, pred_prob = grad_cam.generate_cam(input_tensor, target_class)
    
    # 将CAM覆盖在原始图像上
    visualization = grad_cam.overlay_cam(input_tensor[0], cam)
    
    # 显示原始图像和热力图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    pred_label = class_names[pred_class] if class_names else f"Class {pred_class}"
    plt.title(f'Prediction: {pred_label} ({pred_prob:.4f})')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def visualize_model_activations(model, dataloader, class_names, device, layer_name, num_images=4, save_dir=None):
    """
    可视化模型在多个样本上的激活
    
    Args:
        model: 模型
        dataloader: 数据加载器
        class_names: 类别名称列表
        device: 设备
        layer_name: 要可视化的层名称
        num_images: 要显示的图像数量
        save_dir: 保存目录
    """
    # 设置模型为评估模式
    model.eval()
    
    # 获取一批数据
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 找到目标层
    target_layer = None
    for name, module in model.named_modules():
        if layer_name in name and isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError(f"没有找到名为 {layer_name} 的卷积层")
    
    # 初始化Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # 可视化每个图像
    for i in range(num_images):
        image = images[i:i+1].to(device)
        true_class = labels[i].item()
        
        # 生成CAM
        cam, pred_class, pred_prob = grad_cam.generate_cam(image)
        
        # 将CAM覆盖在原始图像上
        visualization = grad_cam.overlay_cam(image[0], cam)
        
        # 显示原始图像和热力图
        plt.figure(figsize=(12, 5))
        
        # 反归一化图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        orig_img = images[i] * std + mean
        
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img.permute(1, 2, 0).cpu().numpy())
        true_label = class_names[true_class] if class_names else f"Class {true_class}"
        plt.title(f'True: {true_label}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        pred_label = class_names[pred_class] if class_names else f"Class {pred_class}"
        plt.title(f'Pred: {pred_label} ({pred_prob:.4f})')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'cam_image_{i}.png'))
        
        plt.show()
        
        # 可视化特征图
        visualize_features(
            model, 
            layer_name, 
            image, 
            num_features=16, 
            save_path=os.path.join(save_dir, f'features_image_{i}.png') if save_dir else None
        ) 