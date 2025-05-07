import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import argparse
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image

from model import create_alexnet_model, AlexNet
from data_utils import load_caltech101_data
from evaluate import load_model

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
    
def visualize_tsne(model, dataloader, device, class_names, save_path=None):
    """
    使用t-SNE可视化特征空间
    
    参数:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        class_names: 类别名称
        save_path: 保存路径，如果为None则只显示不保存
    """
    try:
        from sklearn.manifold import TSNE
        import seaborn as sns
    except ImportError:
        print("需要安装scikit-learn和seaborn才能使用t-SNE可视化")
        return
        
    # 提取特征
    model.eval()
    features = []
    labels_list = []
    
    # 获取倒数第二层的特征
    def hook_features(module, input, output):
        features.append(input[0].detach().cpu().numpy())
        
    # 注册钩子到最后一个全连接层
    for name, module in model.named_modules():
        if name == 'classifier.6': 
            module.register_forward_hook(hook_features)
    
    # 收集特征和标签
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            labels_list.append(labels.numpy())
    
    # 合并特征和标签
    features = np.vstack(features)
    labels_list = np.concatenate(labels_list)
    
    # 使用t-SNE降维
    print("正在使用t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # 可视化t-SNE结果
    plt.figure(figsize=(10, 8))
    
    # 限制只显示前20个类别，避免图太乱
    num_classes_to_show = min(20, len(class_names))
    mask = labels_list < num_classes_to_show
    
    # 使用seaborn的散点图
    sns.scatterplot(
        x=features_tsne[mask, 0], 
        y=features_tsne[mask, 1],
        hue=[class_names[i] for i in labels_list[mask]],
        palette="tab10",
        alpha=0.8,
        legend="brief"
    )
    
    plt.title('t-SNE特征空间可视化')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')
    plt.legend(loc='best', ncol=2)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()
    
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AlexNet模型可视化')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--results_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--visualize_predictions', action='store_true', help='可视化模型预测')
    parser.add_argument('--visualize_features', action='store_true', help='可视化特征图')
    parser.add_argument('--visualize_tsne', action='store_true', help='可视化t-SNE特征空间')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 加载数据
    train_loader, val_loader, test_loader, class_names = load_caltech101_data(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"类别数量: {len(class_names)}")
    
    # 加载模型
    model = AlexNet(num_classes=len(class_names))
    model = load_model(model, args.model_path, device)
    model = model.to(device)
    print(f"模型加载成功: {args.model_path}")
    
    # 默认执行所有可视化
    do_all = not (args.visualize_predictions or args.visualize_features or args.visualize_tsne)
    
    # 可视化模型预测
    if args.visualize_predictions or do_all:
        print("正在可视化模型预测...")
        visualize_model_predictions(
            model, 
            test_loader, 
            device, 
            class_names, 
            num_images=6, 
            save_path=f"{args.results_dir}/model_predictions.png"
        )
    
    # 可视化特征图
    if args.visualize_features or do_all:
        print("正在可视化特征图...")
        visualize_feature_maps(
            model,
            test_loader,
            device,
            layer_name='features.10',  # 最后一个卷积层
            num_images=1,
            save_path=f"{args.results_dir}/feature_maps.png"
        )
    
    # 可视化t-SNE特征空间
    if args.visualize_tsne or do_all:
        print("正在可视化t-SNE特征空间...")
        visualize_tsne(
            model,
            test_loader,
            device,
            class_names,
            save_path=f"{args.results_dir}/tsne_features.png"
        )
    
    print(f"所有可视化结果已保存到目录: {args.results_dir}")

if __name__ == '__main__':
    main() 