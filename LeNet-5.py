#!/usr/bin/env python
# coding: utf-8

# 导入 pythorch 与绘图相关库
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import FashionMNIST


# 定义 LeNet-5 模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.l1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0
        )
        self.l2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.l3 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.l4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.l5 = nn.Linear(in_features=5 * 5 * 16, out_features=120)
        self.l6 = nn.Linear(in_features=120, out_features=84)
        self.l7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.sigmoid(self.l1(x))
        x = self.l2(x)
        x = self.sigmoid(self.l3(x))
        x = self.l4(x)

        x = self.flatten(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lenet = LeNet().to(device)
print(summary(lenet, (1, 32, 32)))

batch_size = 512
# 加载 FashionMNIST 中的训练数据集
train_data = FashionMNIST(
    root="./data",
    train=True,
    transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]),
    download=True,
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

# 加载测试数据集
test_data = FashionMNIST(
    root="./data",
    train=False,
    transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]),
    download=True,
)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

# 查看训练集中第一批次前5个数据
figure = plt.figure(figsize=(10, 5))
for step, (x, y) in enumerate(train_loader):
    for batch in range(5):
        img = x[batch, :, :].squeeze().numpy()
        label = train_data.classes[y[batch].numpy()]
        figure.subplots_adjust(wspace=1)
        figure.add_subplot(1, 5, batch + 1)
        plt.title(label, size=10)
        plt.imshow(img, cmap=plt.cm.gray)
    break
plt.show()

# 模型训练
optimizer = torch.optim.Adam(lenet.parameters(), lr=0.001)
lossfn = nn.CrossEntropyLoss()
lenet.train()

train_losses_list = []
train_corrects_list = []

for epoch in range(200):
    train_losses = 0.0
    train_corrects = 0.0
    train_nums = 0

    for step, (X, Y) in enumerate(train_loader):
        X = X.to(device)
        Y = Y.to(device)

        o = lenet(X)
        loss = lossfn(o, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses += loss.item() * X.shape[0]
        train_corrects += torch.sum(torch.argmax(o, dim=1) == Y.data)
        train_nums += X.shape[0]

    print(
        f"Epoch {epoch}, train_loss {train_losses / train_nums}, train_corrects {train_corrects / train_nums}"
    )
    train_losses_list.append(train_losses / train_nums)
    train_corrects_list.append(train_corrects / train_nums)

# 模型测试
lenet.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = lenet(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
        
    
plt.plot(train_losses_list)
plt.plot(train_corrects_list)
plt.show()
