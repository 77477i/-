import torch.nn as nn
import torch.nn.functional as F

'''
输入图片大小W×W
Filter大小FXF
步长S
padding的像素数P
'''

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 最后10的输出，需要根据数据进行修改

    def forward(self, x):
        # N=(W-F+2P)/S+1=(32-5)/1+1=28
        x = F.relu(self.conv1(x))       # input(3, 32, 32) output(16, 28, 28)
        # N=(W-F+2P)/S+1=(28-2)/2+1=14
        x = self.pool1(x)               # output(16, 14, 14)
        # N=(W-F+2P)/S+1=(14-5)+1=10
        x = F.relu(self.conv2(x))       # output(32， 10, 10)
        # N=(W-F+2P)/S+1=(10-2)/2+1=5
        x = self.pool2(x)               # output(32， 5, 5)
        # 通过view的函数展平维度，-1代表第一个维度进行自动推理， 32 * 5 * 5
        x = x.view(-1, 32 * 5 * 5)      # output(32 * 5 * 5)
        x = F.relu(self.fc1(x))         # output(120)
        x = F.relu(self.fc2(x))         # output(84)
        x = self.fc3(x)                 # output(10)
        return x


if __name__ == '__main__':
    import torch
    input1 = torch.rand([32,3,32,32])
    model = LeNet()  # 实例化模型
    print(model)
    output = model(input1)
    print(output.shape)












