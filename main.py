import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import subprocess

from data_utils import load_caltech101_data, get_num_classes
from model import create_alexnet_model, get_parameter_groups
from train import train_model, validate_model
from evaluate import evaluate_model, plot_confusion_matrix, generate_classification_report, compare_models, plot_top_misclassifications
from visualize import show_batch, log_model_graph, visualize_model_predictions, log_images_to_tensorboard
from utils import create_optimizer, create_scheduler, Logger
try:
    from tensorboard_exporter import load_tensorboard_data, plot_training_curves
    TENSORBOARD_EXPORTER_AVAILABLE = True
except ImportError:
    TENSORBOARD_EXPORTER_AVAILABLE = False

def export_tensorboard_plots(log_dir, results_dir, logger):
    """
    导出TensorBoard图表到结果目录
    
    Args:
        log_dir: TensorBoard日志目录
        results_dir: 结果保存目录
        logger: 日志记录器
    """
    if not TENSORBOARD_EXPORTER_AVAILABLE:
        logger.warning("未找到tensorboard_exporter模块，无法导出TensorBoard图表")
        logger.info("请运行 'uv run tensorboard_exporter.py --log_dir runs/xxxx --output_dir results/xxxx' 手动导出")
        return
    
    try:
        logger.info("导出TensorBoard图表...")
        data = load_tensorboard_data(log_dir)
        plot_training_curves(data, results_dir)
        logger.info(f"TensorBoard图表已保存至: {results_dir}")
    except Exception as e:
        logger.error(f"导出TensorBoard图表失败: {e}")
        logger.info("请运行 'uv run tensorboard_exporter.py --log_dir runs/xxxx --output_dir results/xxxx' 手动导出")
    
def open_tensorboard(log_dir, port=6006):
    """
    启动TensorBoard服务
    
    Args:
        log_dir: TensorBoard日志目录
        port: TensorBoard服务端口
    
    Returns:
        进程对象
    """
    cmd = f"tensorboard --logdir={log_dir} --port={port}"
    process = subprocess.Popen(cmd, shell=True)
    print(f"TensorBoard已启动，请访问 http://localhost:{port}")
    print(f"按Ctrl+C终止服务")
    return process

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='AlexNet微调 - Caltech101分类')
    
    # 基础参数
    parser.add_argument('--mode', type=str, default='finetune', choices=['finetune', 'scratch', 'evaluate', 'tensorboard'],
                        help='运行模式: finetune, scratch, evaluate, tensorboard')
    parser.add_argument('--data_dir', type=str, default='caltech-101/101_ObjectCategories',
                        help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--feature_extract', action='store_true',
                        help='仅训练最后一层')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径，用于评估模式')
                        
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['sgd', 'adam', 'adamw', 'rmsprop', 'adadelta', 'adagrad'],
                        help='优化器选择')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--finetune_lr', type=float, default=0.0001,
                        help='微调学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='权重衰减')
    
    # 学习率调度参数
    parser.add_argument('--scheduler', type=str, default=None, 
                        choices=['constant', 'warmup', 'linear_decay', 'linear_warmup_decay', 
                                'cosine', 'cosine_warmup', 'step', 'reduce_on_plateau'],
                        help='学习率调度器选择')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='预热步数')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='最小学习率')
    parser.add_argument('--patience', type=int, default=5,
                        help='学习率下降前等待的轮数（用于reduce_on_plateau）')
    
    # 训练高级参数
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='梯度裁剪阈值')
    parser.add_argument('--early_stopping', type=int, default=None,
                        help='早停轮数')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='使用混合精度训练')
    
    # 日志参数
    parser.add_argument('--log_level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='日志级别')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='日志打印频率（每N个批次）')
    parser.add_argument('--save_frequency', type=int, default=1,
                        help='模型保存频率（每N个epoch）')
    
    # TensorBoard参数
    parser.add_argument('--tensorboard_dir', type=str, default=None,
                        help='TensorBoard日志目录，用于启动TensorBoard或导出图表')
    parser.add_argument('--tensorboard_port', type=int, default=6006,
                        help='TensorBoard服务端口')
    parser.add_argument('--export_dir', type=str, default=None,
                        help='TensorBoard图表导出目录')
    
    # 可视化参数
    parser.add_argument('--normalize_cm', action='store_true', default=True,
                        help='是否归一化混淆矩阵')
    parser.add_argument('--max_classes', type=int, default=30,
                        help='混淆矩阵显示的最大类别数量')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # TensorBoard模式
    if args.mode == 'tensorboard':
        if not args.tensorboard_dir:
            print("错误: 请提供--tensorboard_dir参数")
            return
            
        if args.export_dir:
            # 导出TensorBoard图表
            try:
                from tensorboard_exporter import load_tensorboard_data, plot_training_curves
                data = load_tensorboard_data(args.tensorboard_dir)
                plot_training_curves(data, args.export_dir)
                print(f"TensorBoard图表已导出至: {args.export_dir}")
            except Exception as e:
                print(f"导出TensorBoard图表失败: {e}")
        else:
            # 启动TensorBoard服务
            process = open_tensorboard(args.tensorboard_dir, args.tensorboard_port)
            try:
                # 等待用户手动终止
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                process.terminate()
                print("TensorBoard服务已终止")
        
        return
    
    # 检查GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建日志文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'{args.mode}_{timestamp}')
    models_dir = os.path.join('models', f'{args.mode}_{timestamp}')
    log_dir = os.path.join('runs', f'{args.mode}_{timestamp}')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = Logger(log_level=args.log_level, log_file=os.path.join(log_dir, 'main.log'))
    logger.info("=" * 50)
    logger.info(f"开始实验: {args.mode} - {timestamp}")
    logger.info("=" * 50)
    logger.info(f"参数配置:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # 加载数据
    logger.info(f"加载Caltech-101数据集...")
    train_loader, val_loader, class_to_idx = load_caltech101_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # 获取类别数量和类别名称
    num_classes = get_num_classes(args.data_dir)
    logger.info(f"类别数量: {num_classes}")
    
    # 创建类别名称映射
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    if args.mode == 'finetune' or args.mode == 'scratch':
        # 创建模型
        pretrained = (args.mode == 'finetune')
        model = create_alexnet_model(num_classes=num_classes, pretrained=pretrained)
        model = model.to(device)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 配置优化器参数
        params_to_update = get_parameter_groups(
            model, 
            feature_extract=args.feature_extract,
            lr=args.lr,
            finetune_lr=args.finetune_lr
        )
        
        # 创建优化器
        optimizer = create_optimizer(
            args.optimizer, 
            params_to_update, 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # 计算每个epoch的步数
        steps_per_epoch = len(train_loader)
        
        # 创建学习率调度器
        scheduler = None
        if args.scheduler:
            scheduler = create_scheduler(
                args.scheduler,
                optimizer,
                args.epochs,
                steps_per_epoch,
                warmup_steps=args.warmup_steps,
                min_lr=args.min_lr,
                patience=args.patience
            )
        
        # 显示数据样本
        show_batch(
            train_loader, 
            class_names=class_names,
            save_path=os.path.join(results_dir, 'data_samples.png')
        )
        
        # 训练模型
        logger.info(f"开始训练模型...")
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=args.epochs,
            device=device,
            model_save_path=models_dir,
            log_dir=log_dir,
            model_name=args.mode,
            scheduler=scheduler,
            log_frequency=args.log_frequency,
            grad_clip=args.grad_clip,
            save_frequency=args.save_frequency,
            mixed_precision=args.mixed_precision,
            early_stopping=args.early_stopping
        )
        
        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(models_dir, f'{args.mode}_final.pth'))
        
        # 评估最终模型
        logger.info(f"评估最终模型...")
        loss, accuracy, all_preds, all_labels = evaluate_model(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names
        )
        
        # 保存模型信息
        with open(os.path.join(results_dir, 'model_results.txt'), 'w') as f:
            f.write(f"模式: {args.mode}\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"损失: {loss:.4f}\n")
            f.write(f"类别数量: {num_classes}\n")
            f.write(f"批处理大小: {args.batch_size}\n")
            f.write(f"训练轮数: {args.epochs}\n")
            f.write(f"优化器: {args.optimizer}\n")
            f.write(f"学习率: {args.lr}\n")
            f.write(f"微调学习率: {args.finetune_lr}\n")
            f.write(f"权重衰减: {args.weight_decay}\n")
            f.write(f"学习率调度器: {args.scheduler}\n")
            f.write(f"仅训练最后一层: {args.feature_extract}\n")
            f.write(f"混合精度训练: {args.mixed_precision}\n")
        
        # 绘制混淆矩阵
        logger.info(f"绘制混淆矩阵...")
        plot_confusion_matrix(
            true_labels=all_labels,
            predictions=all_preds,
            class_names=class_names,
            save_path=os.path.join(results_dir, 'confusion_matrix.png'),
            normalize=args.normalize_cm,
            max_classes=args.max_classes
        )
        
        # 绘制最常见的错误分类
        logger.info(f"绘制错误分类分析...")
        plot_top_misclassifications(
            true_labels=all_labels,
            predictions=all_preds,
            class_names=class_names,
            top_n=10,
            save_path=os.path.join(results_dir, 'top_misclassifications.png')
        )
        
        # 生成分类报告
        logger.info(f"生成分类报告...")
        generate_classification_report(
            true_labels=all_labels,
            predictions=all_preds,
            class_names=class_names,
            save_path=os.path.join(results_dir, 'classification_report.txt')
        )
        
        # 可视化模型预测
        logger.info(f"可视化模型预测...")
        visualize_model_predictions(
            model=model,
            dataloader=val_loader,
            class_names=class_names,
            device=device,
            save_path=os.path.join(results_dir, 'model_predictions.png')
        )
        
        # 导出TensorBoard图表
        export_tensorboard_plots(log_dir, results_dir, logger)
        
    elif args.mode == 'evaluate':
        # 检查模型路径
        if args.model_path is None or not os.path.exists(args.model_path):
            logger.error(f"错误: 请提供有效的模型路径")
            return
        
        # 创建模型
        model = create_alexnet_model(num_classes=num_classes)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 评估模型
        logger.info(f"评估模型: {args.model_path}")
        loss, accuracy, all_preds, all_labels = evaluate_model(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names
        )
        
        # 保存评估结果
        with open(os.path.join(results_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(f"模型路径: {args.model_path}\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"损失: {loss:.4f}\n")
        
        # 绘制混淆矩阵
        plot_confusion_matrix(
            true_labels=all_labels,
            predictions=all_preds,
            class_names=class_names,
            save_path=os.path.join(results_dir, 'confusion_matrix.png'),
            normalize=args.normalize_cm,
            max_classes=args.max_classes
        )
        
        # 绘制最常见的错误分类
        plot_top_misclassifications(
            true_labels=all_labels,
            predictions=all_preds,
            class_names=class_names,
            top_n=10,
            save_path=os.path.join(results_dir, 'top_misclassifications.png')
        )
        
        # 生成分类报告
        generate_classification_report(
            true_labels=all_labels,
            predictions=all_preds,
            class_names=class_names,
            save_path=os.path.join(results_dir, 'classification_report.txt')
        )
    
    logger.info(f"完成! 结果保存在: {results_dir}")
    logger.info(f"使用以下命令查看TensorBoard: tensorboard --logdir={log_dir}")
    logger.info(f"或运行: uv run main.py --mode tensorboard --tensorboard_dir {log_dir}")

if __name__ == "__main__":
    main() 