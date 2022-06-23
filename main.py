from unittest import result
# 1930200064孙哲渠
import torch
from caffe2.python.trt import transform
from torch.onnx.symbolic_opset9 import view
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import os
from PIL import Image
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),  # 转为tensor，范围改为0-1
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化 加快梯度下降的最优解的速度，即加快训练网络的收敛性；同时也有可能提高精度
])  # 预处理

# 训练数据和测试数据
train_data = MNIST(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
test_data = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, shuffle=False, batch_size=64)
# train_data[0][0].show()  # 展示这个数据的图像，展示的时候需要把transform=transform删掉


class Model(nn.Module):  # 模型
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 10)  # 10个手写数字对应10个输出r

    def forward(self, x):
        x = x.view(-1, 784)  # 变形
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return x


model = Model()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), 0.2)

if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load('./model/model.pkl'))  # 加载保存模型的参数


def train(epoch):
    for index, data in enumerate(train_loader):
        input, target = data  # input为输入数据，target为标签
        optimizer.zero_grad()
        y_predict = model(input)
        loss = criterion(y_predict, target)
        loss.backward()
        optimizer.step()
        if index % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")
            print(loss.item())


def test():
    correct = 0  # 正确的个数
    total = 0  # 总数
    with torch.no_grad():  # 测试不用计算梯度
        for data in test_loader:
            input, target = data
            output = model(input)  # ouput输出10个预测取值，其中最大的几位预测的数
            _, predict = torch.max(output.data, dim=1)  # 返回一个元组，第一个为最大值，第二个为最大值的下表
            total += target.size(0)  # target为形状为(batch_size,1)的矩阵，取size(0)拿出该批的大小
            correct += (predict == target).sum().item()  # predict和target均为(batch_size,1)的矩阵，sum(0)
    print(correct/total)


if __name__ == '__main__':
    # for i in range(2):
    #     train(i)
    #     test()
    img=Image.open("number.png").convert("L")
    img=transform(img) #利用上面的transform进行预处理
    img.view(-1,784)
    result=model(img)
    a,predict=torch.max(result.data,dim=1)
    print(result)
    print(a)
    print("the result is :",predict.item())
    print(("by -1930200064孙哲渠"))