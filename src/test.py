"""
完整的模型测试和评估脚本
包含：详细指标评估 + 推理速度测试 + 业务验证
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.mnist import SimpleCNN
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import random

# 尝试导入 sklearn（用于详细评估）
try:
    from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    print("警告: sklearn 未安装，将使用基础评估方法")
    SKLEARN_AVAILABLE = False

def load_test_data(batch_size: int = 1000):
    """加载测试数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        './data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return test_loader, test_dataset

def basic_accuracy_evaluation(model, test_loader, device):
    """基础准确率评估（不依赖 sklearn）"""
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    return accuracy, all_predictions, all_targets

def detailed_evaluate_with_sklearn(predictions, targets):
    """使用 sklearn 进行详细评估"""
    if not SKLEARN_AVAILABLE:
        return None
        
    # 计算各种指标
    accuracy = 100. * sum(np.array(predictions) == np.array(targets)) / len(targets)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted'
    )
    
    report = classification_report(
        targets, predictions, 
        target_names=[str(i) for i in range(10)],
        digits=4
    )
    
    cm = confusion_matrix(targets, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, save_path='./models/confusion_matrix.png'):
    """绘制混淆矩阵"""
    if not SKLEARN_AVAILABLE:
        print("跳过混淆矩阵绘制（sklearn 未安装）")
        return
        
    plt.figure(figsize=(12, 10))
    labels = [str(i) for i in range(10)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - MNIST CNN', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def measure_inference_speed(model, device, num_samples=1000):
    """测量推理速度"""
    print(f"\n=== 推理速度测试 ===")
    print(f"测试样本数量: {num_samples}")
    
    # 创建随机输入数据
    dummy_input = torch.randn(num_samples, 1, 28, 28).to(device)
    
    # 预热 GPU/CPU
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input[:10])
    
    # 同步 GPU（如果使用）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测量推理时间
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_per_sample = (total_time / num_samples) * 1000  # 转换为毫秒
    throughput = num_samples / total_time  # 样本/秒
    
    print(f"总推理时间: {total_time:.4f} 秒")
    print(f"平均推理延迟: {avg_time_per_sample:.2f} 毫秒/样本")
    print(f"吞吐量: {throughput:.2f} 样本/秒")
    
    return {
        'total_time': total_time,
        'avg_latency_ms': avg_time_per_sample,
        'throughput': throughput
    }

def test_single_image(model, test_dataset, device, idx=None):
    """测试单张图像预测"""
    if idx is None:
        idx = random.randint(0, len(test_dataset) - 1)
    
    image, true_label = test_dataset[idx]
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        predicted = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
    
    # 显示结果
    plt.figure(figsize=(6, 4))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'True: {true_label}, Predicted: {predicted}\nConfidence: {confidence:.2%}')
    plt.axis('off')
    plt.show()
    
    return true_label, predicted, confidence

def business_validation(predictions, targets):
    """业务验证分析"""
    print(f"\n=== 业务验证分析 ===")
    
    # 错误分析
    errors = [(i, pred, true) for i, (pred, true) in enumerate(zip(predictions, targets)) if pred != true]
    error_rate = len(errors) / len(targets) * 100
    
    print(f"错误率: {error_rate:.2f}%")
    print(f"总错误数量: {len(errors)}")
    
    if len(errors) > 0:
        # 找出最常见的错误类型
        error_pairs = [(true, pred) for _, pred, true in errors]
        from collections import Counter
        error_counter = Counter(error_pairs)
        most_common_errors = error_counter.most_common(5)
        
        print("\n最常见的错误类型 (真实标签 -> 预测标签):")
        for (true_label, pred_label), count in most_common_errors:
            print(f"  {true_label} -> {pred_label}: {count} 次")
    
    # 性能基准对比
    print(f"\n=== 性能基准参考 ===")
    print("MNIST 数据集基准性能:")
    print("- 简单 CNN 模型: 98-99% 准确率")
    print("- 复杂模型 (ResNet等): 99.5%+ 准确率")
    print("- 人类水平: ～98% 准确率")
    
    return {
        'error_rate': error_rate,
        'total_errors': len(errors),
        'common_errors': most_common_errors if errors else []
    }

def main():
    """主测试流程"""
    print("开始模型测试和评估...")
    
    # 设备配置
    device = torch.device('cpu')  # 或 'cuda' 如果 GPU 兼容
    print(f"使用设备: {device}")
    
    # 加载模型
    model_path = './models/mnist_cnn.pth'
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
    
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"模型加载成功! 参数数量: {model.count_parameters():,}")
    
    # 加载测试数据
    test_loader, test_dataset = load_test_data()
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 基础准确率评估
    print("\n=== 基础准确率评估 ===")
    accuracy, predictions, targets = basic_accuracy_evaluation(model, test_loader, device)
    print(f"测试集准确率: {accuracy:.2f}%")
    
    # 详细评估（如果 sklearn 可用）
    detailed_metrics = None
    if SKLEARN_AVAILABLE:
        print("\n=== 详细指标评估 ===")
        detailed_metrics = detailed_evaluate_with_sklearn(predictions, targets)
        if detailed_metrics is not None:
            print(f"准确率: {detailed_metrics['accuracy']:.2f}%")
            print(f"精确率: {detailed_metrics['precision']:.4f}")
            print(f"召回率: {detailed_metrics['recall']:.4f}")
            print(f"F1 分数: {detailed_metrics['f1']:.4f}")
            print(f"\n分类报告:\n{detailed_metrics['classification_report']}")
        
            # 绘制混淆矩阵
            plot_confusion_matrix(detailed_metrics['confusion_matrix'])
    else:
        print("\n跳过详细指标评估（sklearn 未安装）")
    
    # 推理速度测试
    speed_metrics = measure_inference_speed(model, device)
    
    # 单张图像测试
    print(f"\n=== 单张图像预测测试 ===")
    true_label, predicted, confidence = test_single_image(model, test_dataset, device)
    print(f"单张测试结果: 真实={true_label}, 预测={predicted}, 置信度={confidence:.2%}")
    
    # 业务验证
    business_metrics = business_validation(predictions, targets)
    
    # 保存评估结果
    results = {
        'basic_accuracy': accuracy,
        'detailed_metrics': detailed_metrics,
        'speed_metrics': speed_metrics,
        'business_metrics': business_metrics
    }
    
    # 打印总结
    print(f"\n{'='*50}")
    print(f"评估总结")
    print(f"{'='*50}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"平均推理延迟: {speed_metrics['avg_latency_ms']:.2f} ms")
    print(f"吞吐量: {speed_metrics['throughput']:.2f} 样本/秒")
    print(f"错误率: {business_metrics['error_rate']:.2f}%")
    print(f"{'='*50}")
    
    return results

if __name__ == "__main__":
    main()