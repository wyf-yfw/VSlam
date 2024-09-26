import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_filenames[idx])
        label_name = os.path.join(self.label_dir, self.image_filenames[idx].replace('.jpg', '.txt'))

        image = Image.open(img_name).convert('RGB')
        with open(label_name, 'r') as f:
            label = int(f.read().strip())  # 假设标签是整数

        if self.transform:
            image = self.transform(image)

        return image, label


# 设置变换，包括归一化
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),  # 转换为灰度图
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # 归一化，灰度图只有一个通道
])

# 创建数据集和数据加载器
train_dataset = CustomDataset(img_dir='C:\\Users\\wyf\\PycharmProjects\\slam\\dataset\\train\\images',
                              label_dir='C:\\Users\\wyf\\PycharmProjects\\slam\\dataset\\train\\labels',
                              transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. 加载 ResNet 模型
model = models.resnet50(pretrained=True)  # 使用预训练模型
model.fc = nn.Linear(model.fc.in_features, 8)  # 修改最后一层以匹配类别数
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

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

print('Training complete.')
