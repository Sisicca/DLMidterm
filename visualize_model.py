import os
import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.models import alexnet, AlexNet_Weights
from data_utils import load_caltech101_data, get_num_classes
from model import create_alexnet_model
from utils import Logger
from advanced_visualization import (
    GradCAM, visualize_features, visualize_prediction_with_gradcam,
    visualize_model_activations, preprocess_image
)

def main():
    parser = argparse.ArgumentParser(description='模型可视化工具')
    
    # 基本参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--data_dir', type=str, default='caltech-101/101_ObjectCategories',
                        help='数据目录，用于加载类别名称')
    parser.add_argument('--image_path', type=str, default=None,
                        help='要可视化的单个图像路径，不设置则使用数据集图像')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批处理大小，仅在使用数据集时有效')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                        help='可视化结果保存目录')
    
    # 可视化选项
    parser.add_argument('--mode', type=str, choices=['gradcam', 'features', 'activations', 'all'],
                        default='all', help='可视化模式')
    parser.add_argument('--target_layer', type=str, default='features.10',
                        help='目标层名称，用于GradCAM和特征可视化')
    parser.add_argument('--num_images', type=int, default=4,
                        help='要可视化的图像数量，仅在使用数据集时有效')
    parser.add_argument('--num_features', type=int, default=16,
                        help='要可视化的特征数量')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = Logger(log_level='info', log_file=os.path.join(args.output_dir, 'visualization.log'))
    logger.info("开始可视化过程...")
    
    # 检查设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载类别信息
    num_classes = get_num_classes(args.data_dir)
    logger.info(f"类别数量: {num_classes}")
    
    # 加载训练数据以获取类别名称
    train_loader, val_loader, class_to_idx = load_caltech101_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    model = create_alexnet_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 找到目标层
    target_layer = None
    for name, module in model.named_modules():
        if args.target_layer in name:
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                logger.info(f"找到目标卷积层: {name}")
                break
    
    if target_layer is None:
        logger.error(f"无法找到指定的目标层: {args.target_layer}")
        return
    
    # 如果提供了单个图像路径
    if args.image_path:
        logger.info(f"处理单个图像: {args.image_path}")
        
        if args.mode in ['gradcam', 'all']:
            logger.info("生成Grad-CAM热力图...")
            visualize_prediction_with_gradcam(
                model=model,
                image_path=args.image_path,
                target_layer=target_layer,
                class_names=class_names,
                device=device,
                save_path=os.path.join(args.output_dir, 'gradcam_single.png')
            )
        
        if args.mode in ['features', 'all']:
            logger.info("可视化特征图...")
            original_image, input_tensor = preprocess_image(args.image_path)
            input_tensor = input_tensor.to(device)
            visualize_features(
                model=model,
                layer_name=args.target_layer,
                input_image=input_tensor,
                num_features=args.num_features,
                save_path=os.path.join(args.output_dir, 'features_single.png')
            )
    else:
        # 使用数据集中的图像
        logger.info(f"使用数据集中的 {args.num_images} 张图像")
        
        if args.mode in ['activations', 'all']:
            logger.info("可视化模型激活...")
            visualize_model_activations(
                model=model,
                dataloader=val_loader,
                class_names=class_names,
                device=device,
                layer_name=args.target_layer,
                num_images=args.num_images,
                save_dir=args.output_dir
            )
    
    logger.info("可视化过程完成")

if __name__ == '__main__':
    main() 