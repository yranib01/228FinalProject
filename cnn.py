import torch
import torch.nn as nn
import torch.optim as optim

from tv_split import *

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(21, 1, 3, bias=True)
        self.lin_final = nn.Linear(7498, 1, bias=True)
        self.activation_final = nn.LeakyReLU()

    def forward(self, x):
        conv_out = self.conv1(x)
        lin_out = self.lin_final(conv_out)
        relu_out = self.activation_final(lin_out)

        return relu_out

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv1d(21, 1, 3, padding='valid')

    def forward(self, x):
        return self.conv1(x)

cnn = MyCNN()

# mynet = TestNet()
# mycnn = MyCNN()
# xtest, ytest = train_data.dataset.x[0], train_data.dataset.y[0]
# xtest, ytest = torch.tensor(xtest).float(), torch.tensor(ytest).float()
# # xtest = torch.transpose(xtest, 1, 2)
# # print(mynet(xtest))
# # print(mynet(xtest).shape)