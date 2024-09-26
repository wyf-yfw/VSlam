import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1. 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder('/Users/wanggang/PycharmProjects/slam/dataset/train', transform=transform)  # 替换为训练数据路径
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. 加载 ResNet 模型
model = models.resnet50(pretrained=True)  # 使用预训练模型
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # 修改最后一层以匹配类别数
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # 移动到 GPU

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
num_epochs = 10  # 设置训练轮数
for epoch in range(num_epochs):
    model.train()  # 训练模式
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

print('Training complete.')
