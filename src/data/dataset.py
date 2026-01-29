"""
数据集加载和预处理模块
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size: int = 64, 
                         train: bool = True,
                         download: bool = True):
    """
    获取 MNIST 数据加载器
    
    Args:
        batch_size: 批次大小
        train: 是否加载训练集
        download: 是否下载数据集
    
    Returns:
        DataLoader 对象
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 统计值
    ])
    
    # 加载数据集
    dataset = datasets.MNIST(
        './data', 
        train=train, 
        download=download, 
        transform=transform
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train
    )
    
    return dataloader