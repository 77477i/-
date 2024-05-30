import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import time

'''将一个PIL图像/NumPy数组转换为PyTorch张量,将图像的像素值从0-255的范围归一化到0-1的范围
   标准化张量图像，标准化的公式是：(输入 - mean) / std。所有三个通道（RGB）的均值和标准差都是0.5，
   输入张量的值已经在0-1之间，所以这种标准化将会把它们变换到-1~1的范围'''
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 50000张训练图片，10000张测试图片
# dataset用于表示数据集，包括如何获取和处理数据的逻辑
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform)

# dataloader用于批量加载数据，提供打乱、并行处理等功能，便于模型训练和测试
trainloader = DataLoader(trainset, batch_size=36, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)


test_dataiter = iter(testloader)                # 将testloader转化为可迭代的迭代器
test_image, test_label = next(test_dataiter)   # 通过next方法可以获得一批数据，数据包括测试图像，标签

# 导入数据标签
classes = ('plane', 'car', 'bird', 'cat', 'deer'
           'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(img):
#     img = img / 2 + 0.5                             # unnormalize，将图像数据从归一化状态还原
#     npimg = img.numpy()                             # 将 PyTorch 张量转换为 NumPy 数组
#     plt.imshow(np.transpose(npimg, (1, 2, 0))) # 将图像的形状从(C, H, W) 转换为(H, W, C)
#     plt.show()
#
# # print labels
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# # show images
# imshow(torchvision.utils.make_grid(test_image))

net = LeNet()                           # 实例化模型
loss_function = nn.CrossEntropyLoss()   #已经包含了softmax函数，不需要在网络输出加上softmax
optimizer = optim.Adam(net.parameters(), lr=0.001)




for epoch in range(5):                                  #一个epoch即表示要将训练集迭代多少轮
    runing_loss = 0.0                                   # 累加在训练过程的损失
    time_start = time.perf_counter()
    for step, data in enumerate(trainloader, start=0):  # 遍历训练集样本，step从0开始计算
        inputs, labels = data
        optimizer.zero_grad()                           # 清除历史梯度

        # forward + backward + optimize
        outputs = net(inputs)                           # 正向传播
        loss = loss_function(outputs, labels)           # 计算损失
        loss.backward()                                 # 反向传播
        optimizer.step()                                # 优化器更新参数


        # 打印耗时、损失、准确率等数据
        runing_loss += loss.item()
        if step % 500 == 499:                              # print every 500 mini-batches，每500步打印一次
            with torch.no_grad():                           # 在以下步骤中（验证过程中）不用计算每个节点的损失梯度，防止内存占用
                outputs = net(test_image)                   # 测试集传入网络（test_batch_size=10000），output维度为[10000,10]
                predict_y = torch.max(outputs, dim=1)[1]    # 以output中值最大位置对应的索引（标签）作为预测输出
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                      (epoch + 1, step + 1, runing_loss / 500, accuracy))

                print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
                runing_loss =0.0

print('Finished Training')

# 保存训练得到的参数
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)