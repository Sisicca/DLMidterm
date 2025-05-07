import torch
import torch.nn as nn
import torchvision.models as models

def create_alexnet_model(num_classes=101, pretrained=True):
    """
    创建AlexNet模型
    
    参数:
        num_classes (int): 类别数量
        pretrained (bool): 是否使用预训练权重
        
    返回:
        model: 修改后的AlexNet模型
    """
    if pretrained:
        # 加载预训练的AlexNet模型
        model = models.alexnet(pretrained=True)
        print("加载预训练的AlexNet模型")
        
        # 冻结所有参数（用于微调阶段）
        for param in model.parameters():
            param.requires_grad = False
    else:
        # 从头开始训练的AlexNet模型
        model = models.alexnet(pretrained=False)
        print("创建未预训练的AlexNet模型")
    
    # 修改最后一个全连接层以适应Caltech-101数据集
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    
    return model

def get_fine_tuning_parameters(model, feature_extract=True):
    """
    获取需要优化的参数
    
    参数:
        model: 模型
        feature_extract (bool): 是否只训练最后一层
        
    返回:
        params_to_update: 需要更新的参数列表
    """
    params_to_update = []
    
    if feature_extract:
        # 只训练最后一层
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
    else:
        # 训练所有参数但使用不同的学习率
        # 分为两组：特征提取部分和分类器部分
        feature_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name:
                if 'classifier.6' in name:  # 输出层（最后一层）
                    classifier_params.append(param)
                else:  # 其他分类器层
                    param.requires_grad = True
                    feature_params.append(param)
            else:  # 特征提取部分
                param.requires_grad = True
                feature_params.append(param)
        
        # 返回分组参数，便于设置不同的学习率
        return [
            {'params': feature_params, 'lr_mult': 0.1},  # 特征提取部分使用较小的学习率
            {'params': classifier_params, 'lr_mult': 1.0}  # 分类器部分使用正常学习率
        ]
    
    return params_to_update 