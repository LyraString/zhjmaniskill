#!/usr/bin/env python3
"""
实时可视化训练曲线
用法: python plot_training.py <logs.json.txt路径>
例如: python plot_training.py data/outputs/2026.01.29/*/logs.json.txt
"""

import sys
import json
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import time

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def read_json_logs(log_path):
    """读取 JSON 日志文件"""
    data = {
        'epoch': [],
        'global_step': [],
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'train_action_mse_error': []
    }
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        for key in data.keys():
                            if key in entry:
                                data[key].append(entry[key])
                            elif len(data[key]) > 0:
                                data[key].append(data[key][-1])  # 保持长度一致
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        print(f"日志文件未找到: {log_path}")
        return None
    
    # 确保所有列表长度一致
    min_len = min(len(v) for v in data.values() if len(v) > 0)
    for key in data:
        data[key] = data[key][:min_len]
    
    return data

def smooth_curve(values, weight=0.9):
    """平滑曲线"""
    if len(values) == 0:
        return []
    smoothed = []
    last = values[0]
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_training_curves(data, save_path=None):
    """绘制训练曲线 - 每个指标独立显示"""
    fig = plt.figure(figsize=(18, 10))
    
    # 训练损失（最重要的指标）
    if len(data['train_loss']) > 0:
        ax1 = plt.subplot(2, 3, 1)
        epochs = data['epoch']
        train_loss = data['train_loss']
        
        # 原始曲线
        ax1.plot(epochs, train_loss, label='原始', color='lightblue', alpha=0.5, linewidth=1)
        # 平滑曲线
        smoothed = smooth_curve(train_loss, weight=0.9)
        ax1.plot(epochs, smoothed, label='平滑', color='blue', linewidth=2)
        
        # 标注最新值
        if len(train_loss) > 0:
            latest_val = train_loss[-1]
            ax1.plot(epochs[-1], latest_val, 'ro', markersize=8)
            ax1.text(epochs[-1], latest_val, f' {latest_val:.4f}', 
                    fontsize=10, va='center', fontweight='bold')
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title(f'训练损失 (当前: {train_loss[-1]:.4f})', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 验证损失
    if len(data['val_loss']) > 0 and any(v > 0 for v in data['val_loss']):
        ax2 = plt.subplot(2, 3, 2)
        val_loss = [v for v in data['val_loss'] if v > 0]
        val_epochs = [data['epoch'][i] for i, v in enumerate(data['val_loss']) if v > 0]
        
        ax2.plot(val_epochs, val_loss, 'o-', color='orange', linewidth=2, markersize=4)
        
        if len(val_loss) > 0:
            latest_val = val_loss[-1]
            ax2.plot(val_epochs[-1], latest_val, 'ro', markersize=8)
            ax2.text(val_epochs[-1], latest_val, f' {latest_val:.4f}', 
                    fontsize=10, va='center', fontweight='bold')
        
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.set_title(f'验证损失 (当前: {val_loss[-1]:.4f})', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 学习率
    if len(data['lr']) > 0:
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(data['global_step'], data['lr'], color='green', linewidth=2)
        
        if len(data['lr']) > 0:
            latest_lr = data['lr'][-1]
            ax3.plot(data['global_step'][-1], latest_lr, 'ro', markersize=8)
            ax3.text(data['global_step'][-1], latest_lr, f' {latest_lr:.2e}', 
                    fontsize=10, va='center', fontweight='bold')
        
        ax3.set_xlabel('Global Step', fontsize=11)
        ax3.set_ylabel('Learning Rate', fontsize=11)
        ax3.set_title(f'学习率 (当前: {data["lr"][-1]:.2e})', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # 动作 MSE 误差
    if len(data['train_action_mse_error']) > 0:
        ax4 = plt.subplot(2, 3, 4)
        mse_data = [v for v in data['train_action_mse_error'] if v > 0]
        mse_epochs = [data['epoch'][i] for i, v in enumerate(data['train_action_mse_error']) if v > 0]
        
        if len(mse_data) > 0:
            ax4.plot(mse_epochs, mse_data, 'o-', color='red', linewidth=2, markersize=4)
            
            latest_mse = mse_data[-1]
            ax4.plot(mse_epochs[-1], latest_mse, 'ro', markersize=8)
            ax4.text(mse_epochs[-1], latest_mse, f' {latest_mse:.4f}', 
                    fontsize=10, va='center', fontweight='bold')
            
            ax4.set_xlabel('Epoch', fontsize=11)
            ax4.set_ylabel('MSE Error', fontsize=11)
            ax4.set_title(f'动作 MSE 误差 (当前: {latest_mse:.4f})', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 训练损失 vs 步数（详细视图）
    if len(data['train_loss']) > 0:
        ax5 = plt.subplot(2, 3, 5)
        steps = data['global_step']
        train_loss_steps = data['train_loss']
        
        ax5.plot(steps, train_loss_steps, color='purple', alpha=0.4, linewidth=1)
        smoothed = smooth_curve(train_loss_steps, weight=0.95)
        ax5.plot(steps, smoothed, color='purple', linewidth=2)
        
        if len(train_loss_steps) > 0:
            latest_val = train_loss_steps[-1]
            ax5.plot(steps[-1], latest_val, 'ro', markersize=8)
        
        ax5.set_xlabel('Global Step', fontsize=11)
        ax5.set_ylabel('Train Loss', fontsize=11)
        ax5.set_title('训练损失（步数视图）', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, linestyle='--')
    
    # 统计信息面板
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = "📊 训练统计\n" + "="*30 + "\n\n"
    if len(data['epoch']) > 0:
        stats_text += f"当前 Epoch: {data['epoch'][-1]}\n"
        stats_text += f"总步数: {data['global_step'][-1]}\n\n"
    
    if len(data['train_loss']) > 0:
        stats_text += f"训练损失:\n"
        stats_text += f"  当前: {data['train_loss'][-1]:.6f}\n"
        stats_text += f"  最小: {min(data['train_loss']):.6f}\n"
        stats_text += f"  最大: {max(data['train_loss']):.6f}\n\n"
    
    if len(data['val_loss']) > 0:
        val_losses = [v for v in data['val_loss'] if v > 0]
        if val_losses:
            stats_text += f"验证损失:\n"
            stats_text += f"  当前: {val_losses[-1]:.6f}\n"
            stats_text += f"  最小: {min(val_losses):.6f}\n\n"
    
    if len(data['lr']) > 0:
        stats_text += f"学习率: {data['lr'][-1]:.2e}\n\n"
    
    if len(data['train_action_mse_error']) > 0:
        mse_vals = [v for v in data['train_action_mse_error'] if v > 0]
        if mse_vals:
            stats_text += f"动作 MSE: {mse_vals[-1]:.6f}\n"
    
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('训练监控面板', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()

def watch_mode(log_path, interval=5):
    """监控模式：每隔几秒刷新一次"""
    print(f"监控模式启动，每 {interval} 秒刷新一次...")
    print(f"监控文件: {log_path}")
    print("按 Ctrl+C 退出\n")
    
    plt.ion()  # 交互模式
    fig = None
    
    try:
        while True:
            data = read_json_logs(log_path)
            if data and len(data['epoch']) > 0:
                if fig:
                    plt.close(fig)
                
                plot_training_curves(data)
                plt.pause(0.1)
                
                # 打印最新状态
                print(f"\r[Epoch {data['epoch'][-1]}] "
                      f"Train Loss: {data['train_loss'][-1]:.4f} | "
                      f"Steps: {data['global_step'][-1]}", end='', flush=True)
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        plt.ioff()
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("用法: python plot_training.py <日志文件路径> [--watch]")
        print("例如: python plot_training.py data/outputs/2026.01.29/*/logs.json.txt")
        print("      python plot_training.py data/outputs/2026.01.29/*/logs.json.txt --watch")
        
        # 尝试自动查找最新的日志文件
        output_dir = Path("data/outputs")
        if output_dir.exists():
            log_files = sorted(output_dir.glob("*/*/logs.json.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if log_files:
                print(f"\n找到最新的日志文件: {log_files[0]}")
                sys.argv.append(str(log_files[0]))
            else:
                sys.exit(1)
        else:
            sys.exit(1)
    
    log_path = sys.argv[1]
    watch = '--watch' in sys.argv or '-w' in sys.argv
    
    if watch:
        watch_mode(log_path, interval=5)
    else:
        data = read_json_logs(log_path)
        if data and len(data['epoch']) > 0:
            output_path = Path(log_path).parent / "training_curves.png"
            plot_training_curves(data, save_path=str(output_path))
            print(f"\n最新状态:")
            print(f"  Epoch: {data['epoch'][-1]}")
            print(f"  Train Loss: {data['train_loss'][-1]:.4f}")
            if data['val_loss']:
                print(f"  Val Loss: {data['val_loss'][-1]:.4f}")
            print(f"  Global Step: {data['global_step'][-1]}")
        else:
            print("未找到有效的训练数据")

if __name__ == "__main__":
    main()
