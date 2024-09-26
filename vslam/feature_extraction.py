import numpy as np
import torch.nn as nn
import cv2
import torch


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入通道为1，输出通道为2
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu')
        self.laplacian = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False, device=self.device)
        # 输入通道改为2，输出通道为1
        self.sharpen1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False, device=self.device)
        self.sharpen2 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False, device=self.device)

        self.laplacian_kernel = torch.tensor([[[0, 1, 0],
                                               [1, -4, 1],
                                               [0, 1, 0]],
                                              [[0, 1, 0],
                                               [1, -4, 1],
                                               [0, 1, 0]]], dtype=torch.float32, device=self.device).unsqueeze(1)  # shape (2, 1, 3, 3)

        self.sharpen_kernel1 = torch.tensor([[[[-1.0, -1.0, -1.0],
                                               [-1.0, -2.0, -1.0],
                                               [-1.0, -1.0, -1.0]],
                                              [[0.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 0.0, 0.0]]]], dtype=torch.float32, device=self.device)
        self.sharpen_kernel2 = torch.tensor([[[[1.0, 1.0, 1.0],
                                               [1.0, 2.0, 1.0],
                                               [1.0, 1.0, 1.0]],
                                              [[0.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 0.0, 0.0]]]], dtype=torch.float32, device=self.device)

        # 赋值给卷积层的权重
        self.laplacian.weight.data = self.laplacian_kernel
        self.sharpen1.weight.data = self.sharpen_kernel1
        self.sharpen2.weight.data = self.sharpen_kernel2

    def conv(self, img):
        # 确保输入图像为浮点数并添加批次和通道维度
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(self.device)
        img = self.laplacian(img)
        img1 = self.sharpen1(img)
        img2 = self.sharpen2(img)

        img1 = img1.squeeze(0).squeeze(0).cpu().detach().numpy()
        img2 = img2.squeeze(0).squeeze(0).cpu().detach().numpy()

        img = np.maximum(img1, img2)
        # 去掉批次和通道维度并转换为 NumPy 数组
        return img


if __name__ == '__main__':
    FE = FeatureExtractor()
    image = cv2.imread('../data/00023.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', image)
    image = FE.conv(image)
    _, image = cv2.threshold(image, 0.8, 1, cv2.THRESH_BINARY)
    cv2.imshow('conv', image)
    cv2.waitKey(0)
