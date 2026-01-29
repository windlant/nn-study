"""
MNIST CNN 模型定义
"""
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """简单的 CNN 模型用于 MNIST 分类"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 输入: 1x28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=0),  # 输出: 32x26x26
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 输出: 32x13x13
            
            nn.Conv2d(32, 64, kernel_size=3, padding=0), # 输出: 64x11x11  
            nn.ReLU(),
            nn.MaxPool2d(2)                             # 输出: 64x5x5
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平: (batch_size, 64*5*5)
        x = self.fc_layers(x)
        return x

    def count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# 实例化模型并打印参数数量
if __name__ == "__main__":
    model = SimpleCNN()
    print(f"模型参数数量: {model.count_parameters():,}")