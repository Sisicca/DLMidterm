import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def setup_style():
    """设置Matplotlib样式"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 2.5
    
    # 使用中文字体 (如果可用)
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    except:
        pass

def load_tensorboard_data(log_dir, size_guidance=None):
    """
    加载TensorBoard日志数据
    
    Args:
        log_dir: TensorBoard日志目录
        size_guidance: 大小限制指南，用于限制加载的事件数量
        
    Returns:
        包含所有标量数据的字典
    """
    if size_guidance is None:
        size_guidance = {
            'scalars': 0,  # 0表示加载所有值
            'histograms': 0,
            'images': 0,
            'audio': 0,
            'tensors': 0,
        }
    
    event_acc = EventAccumulator(log_dir, size_guidance)
    event_acc.Reload()
    
    # 获取所有可用的标量标签
    scalar_tags = event_acc.Tags()['scalars']
    
    # 提取所有标量数据
    data = defaultdict(list)
    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append({
                'step': event.step,
                'wall_time': event.wall_time,
                'value': event.value
            })
    
    # 转换为DataFrame
    for tag in data:
        data[tag] = pd.DataFrame(data[tag])
    
    return dict(data)

def smooth_curve(values, factor=0.9):
    """
    使用指数移动平均对数据进行平滑处理
    
    Args:
        values: 数据值列表
        factor: 平滑因子，值越大平滑程度越高
        
    Returns:
        平滑后的数据
    """
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * factor + (1 - factor) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_multiple_metrics(data, metrics, title, filename, smooth=0.9, legend_loc='best'):
    """
    绘制多个指标的曲线
    
    Args:
        data: 包含标量数据的字典
        metrics: 要绘制的指标列表
        title: 图表标题
        filename: 保存图表的文件名
        smooth: 平滑因子
        legend_loc: 图例位置
    """
    plt.figure(figsize=(12, 8))
    
    for metric in metrics:
        if metric in data:
            steps = data[metric]['step'].values
            values = data[metric]['value'].values
            
            # 绘制原始数据（透明度低）
            plt.plot(steps, values, alpha=0.3, label=f"{metric} (原始)")
            
            # 绘制平滑后的数据
            if smooth > 0:
                smoothed_values = smooth_curve(values, smooth)
                plt.plot(steps, smoothed_values, label=f"{metric} (平滑)")
    
    plt.title(title)
    plt.xlabel('步数')
    plt.ylabel('值')
    plt.legend(loc=legend_loc)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_rate(data, output_dir):
    """绘制学习率变化曲线"""
    if 'learning_rate' not in data:
        return
    
    plt.figure(figsize=(12, 6))
    steps = data['learning_rate']['step'].values
    values = data['learning_rate']['value'].values
    
    plt.plot(steps, values)
    plt.title('学习率变化曲线')
    plt.xlabel('步数')
    plt.ylabel('学习率')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_grad_norm(data, output_dir):
    """绘制梯度范数变化曲线"""
    # 查找所有梯度范数相关的标签
    grad_tags = [tag for tag in data.keys() if 'grad_norm' in tag]
    
    if not grad_tags:
        return
    
    plt.figure(figsize=(12, 6))
    
    for tag in grad_tags:
        steps = data[tag]['step'].values
        values = data[tag]['value'].values
        
        # 对梯度范数进行平滑处理
        smoothed_values = smooth_curve(values, 0.9)
        plt.plot(steps, smoothed_values, label=tag)
    
    plt.title('梯度范数变化曲线')
    plt.xlabel('步数')
    plt.ylabel('梯度范数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grad_norm.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(data, output_dir):
    """
    绘制训练曲线并保存为图表
    
    Args:
        data: 包含标量数据的字典
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置样式
    setup_style()
    
    # 绘制损失曲线
    loss_metrics = [tag for tag in data.keys() if 'loss' in tag.lower()]
    if loss_metrics:
        plot_multiple_metrics(
            data, 
            loss_metrics, 
            '训练和验证损失', 
            os.path.join(output_dir, 'loss_curves.png'),
            smooth=0.9
        )
    
    # 绘制准确率曲线
    acc_metrics = [tag for tag in data.keys() if 'acc' in tag.lower() or 'accuracy' in tag.lower()]
    if acc_metrics:
        plot_multiple_metrics(
            data, 
            acc_metrics, 
            '训练和验证准确率', 
            os.path.join(output_dir, 'accuracy_curves.png'),
            smooth=0.7
        )
    
    # 绘制学习率曲线
    plot_learning_rate(data, output_dir)
    
    # 绘制梯度范数曲线
    plot_grad_norm(data, output_dir)
    
    # 如果存在其他有用的指标，也可以绘制
    # 例如，F1分数、精确率、召回率等
    other_metrics = ['f1', 'precision', 'recall']
    other_metric_tags = []
    for metric in other_metrics:
        other_metric_tags.extend([tag for tag in data.keys() if metric in tag.lower()])
    
    if other_metric_tags:
        plot_multiple_metrics(
            data, 
            other_metric_tags, 
            '其他评价指标', 
            os.path.join(output_dir, 'other_metrics.png'),
            smooth=0.7
        )
    
    print(f"已生成以下图表:")
    print(f" - 损失曲线: {os.path.join(output_dir, 'loss_curves.png')}")
    print(f" - 准确率曲线: {os.path.join(output_dir, 'accuracy_curves.png')}")
    print(f" - 学习率曲线: {os.path.join(output_dir, 'learning_rate.png')}")
    print(f" - 梯度范数曲线: {os.path.join(output_dir, 'grad_norm.png')}")
    if other_metric_tags:
        print(f" - 其他指标: {os.path.join(output_dir, 'other_metrics.png')}")

def main():
    parser = argparse.ArgumentParser(description='从TensorBoard日志导出图表')
    parser.add_argument('--log_dir', type=str, required=True, 
                        help='TensorBoard日志目录')
    parser.add_argument('--output_dir', type=str, default='./tensorboard_exports', 
                        help='图表输出目录')
    parser.add_argument('--smooth', type=float, default=0.9, 
                        help='平滑因子 (0-1)')
    args = parser.parse_args()
    
    print(f"正在从 {args.log_dir} 加载TensorBoard数据...")
    data = load_tensorboard_data(args.log_dir)
    
    print(f"发现 {len(data)} 个指标:")
    for tag in data.keys():
        print(f" - {tag}: {len(data[tag])} 个数据点")
    
    print(f"正在生成图表...")
    plot_training_curves(data, args.output_dir)
    
    print(f"完成! 所有图表已保存至 {args.output_dir}")

if __name__ == "__main__":
    main() 