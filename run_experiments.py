import os
import subprocess
import argparse
from datetime import datetime

def run_command(command):
    """运行命令并打印输出"""
    print(f"执行命令: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print(output.decode('utf-8').strip())
    
    return process.poll()

def main():
    parser = argparse.ArgumentParser(description='运行Caltech-101分类实验')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--finetune_lr', type=float, default=0.0001,
                        help='微调学习率')
    parser.add_argument('--run_all', action='store_true',
                        help='运行所有实验')
    parser.add_argument('--run_finetune', action='store_true',
                        help='运行微调实验')
    parser.add_argument('--run_scratch', action='store_true',
                        help='运行从零训练实验')
    parser.add_argument('--run_feature_extract', action='store_true',
                        help='运行特征提取实验')
    
    args = parser.parse_args()
    
    # 记录实验时间
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"实验开始时间: {timestamp}")
    
    # 创建实验结果目录
    experiment_dir = os.path.join('experiments', timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 记录实验参数
    with open(os.path.join(experiment_dir, 'experiment_params.txt'), 'w') as f:
        f.write(f"实验时间: {timestamp}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"批处理大小: {args.batch_size}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"微调学习率: {args.finetune_lr}\n")
    
    # 定义实验
    experiments = []
    
    if args.run_all or args.run_finetune:
        experiments.append(
            f"python main.py --mode finetune --epochs {args.epochs} "
            f"--batch_size {args.batch_size} --lr {args.lr} "
            f"--finetune_lr {args.finetune_lr}"
        )
    
    if args.run_all or args.run_scratch:
        experiments.append(
            f"python main.py --mode scratch --epochs {args.epochs} "
            f"--batch_size {args.batch_size} --lr {args.lr}"
        )
    
    if args.run_all or args.run_feature_extract:
        experiments.append(
            f"python main.py --mode finetune --feature_extract "
            f"--epochs {args.epochs} --batch_size {args.batch_size} "
            f"--lr {args.lr}"
        )
    
    # 如果没有指定任何实验，默认运行微调实验
    if not experiments:
        experiments.append(
            f"python main.py --mode finetune --epochs {args.epochs} "
            f"--batch_size {args.batch_size} --lr {args.lr} "
            f"--finetune_lr {args.finetune_lr}"
        )
    
    # 运行所有实验
    results = {}
    for i, command in enumerate(experiments):
        print(f"\n{'='*50}")
        print(f"实验 {i+1}/{len(experiments)}")
        print(f"{'='*50}\n")
        
        start_time = datetime.now()
        exit_code = run_command(command)
        end_time = datetime.now()
        
        results[command] = {
            'exit_code': exit_code,
            'duration': (end_time - start_time).total_seconds() / 60.0  # 分钟
        }
    
    # 打印实验结果摘要
    print("\n\n实验结果摘要:")
    print("="*50)
    for command, result in results.items():
        status = "成功" if result['exit_code'] == 0 else f"失败 (代码: {result['exit_code']})"
        print(f"命令: {command}")
        print(f"状态: {status}")
        print(f"耗时: {result['duration']:.2f} 分钟")
        print("-"*50)

if __name__ == "__main__":
    main() 