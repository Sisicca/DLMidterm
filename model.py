import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, num_classes=101):
        """
        AlexNet模型
        
        参数:
            num_classes: 类别数量
        """
        super(AlexNet, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 分类器层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def create_alexnet_model(num_classes=101, pretrained=True):
    """
    创建AlexNet模型
    
    参数:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
    
    返回:
        model: AlexNet模型
    """
    if pretrained:
        # 加载预训练的AlexNet模型
        model = models.alexnet(pretrained=True)
        
        # 修改最后一层以适应新的类别数量
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        # 创建新的AlexNet模型
        model = AlexNet(num_classes=num_classes)
    
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