import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights

def create_alexnet_model(num_classes=101, pretrained=True):
    """
    创建AlexNet模型
    
    Args:
        num_classes: 输出类别数量
        pretrained: 是否使用预训练权重
        
    Returns:
        model: 修改后的AlexNet模型
    """
    if pretrained:
        # 加载在ImageNet上预训练的AlexNet模型
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        print("加载预训练的AlexNet模型")
    else:
        # 从头开始创建AlexNet模型
        model = alexnet(weights=None)
        print("创建随机初始化的AlexNet模型")
    
    # 修改最后的分类层以适应101个类别
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    
    return model

def get_parameter_groups(model, feature_extract=False, lr=0.001, finetune_lr=0.0001):
    """
    为不同层设置不同的学习率
    
    Args:
        model: 模型
        feature_extract: 是否只训练输出层
        lr: 输出层学习率
        finetune_lr: 微调学习率
        
    Returns:
        params_to_update: 参数列表
    """
    # 初始化要更新的参数列表
    params_to_update = []
    
    if feature_extract:
        # 仅更新最后一层
        for param in model.parameters():
            param.requires_grad = False
            
        # 只启用分类器最后一层的梯度更新
        for param in model.classifier[6].parameters():
            param.requires_grad = True
            params_to_update.append(param)
        
        return [{'params': params_to_update, 'lr': lr}]
    else:
        # 微调模式：不同层使用不同的学习率
        feature_params = []
        classifier_params = []
        
        # 分离特征提取器和分类器的参数
        for name, param in model.named_parameters():
            if 'classifier.6' in name:  # 最后一层使用较高的学习率
                param.requires_grad = True
                classifier_params.append(param)
            else:
                param.requires_grad = True  # 允许所有层都可训练
                feature_params.append(param)
        
        # 返回不同学习率的参数组
        return [
            {'params': feature_params, 'lr': finetune_lr},
            {'params': classifier_params, 'lr': lr}
        ] 