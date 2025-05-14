import torch
import torch.nn as nn
import torch.optim as optim

from tv_split import *

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(21, 1, 1500, bias=True)
        self.lin_final = nn.Linear(6001, 1, bias=True)
        # self.activation_final = nn.LeakyReLU()

    def forward(self, x):
        conv_out = self.conv1(x)
        lin_out = self.lin_final(conv_out)
        # relu_out = self.activation_final(lin_out)

        return lin_out

class BigCNN(nn.Module):
    def __init__(self):
        super(BigCNN, self).__init__()
        self.conv1 = nn.Conv1d(21, 10, 1500, stride=100, bias=True)
        self.mp = nn.MaxPool1d(4, stride=3)
        self.conv1_activation = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(10, 1, 10, bias=True)
        self.conv2_activation = nn.LeakyReLU()
        self.lin_final = nn.Linear(7, 1, bias=True)
        # self.activation_final = nn.LeakyReLU()

    def forward(self, x):
        conv_out = self.conv1(x)
        mp_out = self.conv1_activation(self.mp(conv_out))
        conv2_out = self.conv2_activation(self.conv2(mp_out))
        lin_out = self.lin_final(conv2_out)

        return lin_out



# mynet = TestNet()
# mycnn = MyCNN()
# xtest, ytest = train_data.dataset.x[0], train_data.dataset.y[0]
# xtest, ytest = torch.tensor(xtest).float(), torch.tensor(ytest).float()
# # xtest = torch.transpose(xtest, 1, 2)
# # print(mynet(xtest))
# # print(mynet(xtest).shape)