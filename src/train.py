"""
模型训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.mnist import SimpleCNN
from src.data.dataset import get_mnist_dataloaders
import os

def train_model():
    """训练 CNN 模型"""
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型目录
    os.makedirs('./models', exist_ok=True)
    
    # 加载数据
    train_loader = get_mnist_dataloaders(batch_size=64, train=True)
    
    # 初始化模型
    model = SimpleCNN().to(device)
    print(f"模型参数数量: {model.count_parameters():,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    print("开始训练...")
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} 完成，平均损失: {avg_loss:.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), './models/mnist_cnn.pth')
    print("模型已保存到 ./models/mnist_cnn.pth")
    
    return model

if __name__ == "__main__":
    train_model()