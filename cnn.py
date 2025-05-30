import torch
import torch.nn as nn
import torch.optim as optim

# from tv_split import *

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(21, 1, 1500, bias=True)
        # Filter num = 1, Filter size = 1500, stride = 1
        self.lin_final = nn.Linear(6001, 1, bias=True) # 6001 = (7500 + 2*p - 1500(filter_size))/1(stride) + 1
        # self.activation_final = nn.LeakyReLU()

    def forward(self, x):
        conv_out = self.conv1(x)
        lin_out = self.lin_final(conv_out)
        # relu_out = self.activation_final(lin_out)

        return lin_out

class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(21, 10, 1500, stride=100, bias=True) # 61 = (7500 + 2*p - 1500)/100 + 1 / Input: (21, 7500) => (10,61)
        # Filter num = 10, Filter size = 1500, stride = 100 
        self.conv1_activation = nn.LeakyReLU() # => (10, 20)
        self.mp = nn.MaxPool1d(4, stride=3) # Kernel size 4(), stride 3 / (61 + 2*p(0) - 4)/3 + 1 = 20  => (10,20)
        self.conv2 = nn.Conv1d(10, 1, 10, bias=True) # input size = 10, output size = 1, filter size = 10 / 11 = (20-10)/1 + 1 / => (1,11)
        self.conv2_activation = nn.LeakyReLU()
        self.lin_final = nn.Linear(11, 1, bias=True)
        # self.activation_final = nn.LeakyReLU()

    def forward(self, x):
        conv_out = self.conv1(x)
        act_out = self.conv1_activation(conv_out)
        mp_out = self.mp(act_out)
        conv2_out = self.conv2_activation(self.conv2(mp_out)).squeeze(1) # (1,11) => (11)
        lin_out = self.lin_final(conv2_out).squeeze(-1) # (11) => (1) 

        return lin_out

class DeepCNN(nn.Module):
    def __init__(self, activation=nn.LeakyReLU()):
        super().__init__()
        self.conv1 = nn.Conv1d(21, 10, 1500, stride=100, bias=True) # 61 = (7500 + 2*p - 1500)/100 + 1 / Input: (21, 7500) => (10,61)
        # Filter num = 10, Filter size = 1500, stride = 100
        self.mp = nn.MaxPool1d(3, stride=2) # Kernel size 4(), stride 3 / (61 + 2*p(0) - 4)/3 + 1 = 20  => (10,20)
        self.conv2 = nn.Conv1d(10, 5, 10, bias=True)

        self.conv3 = nn.Conv1d(5, 5, 10, bias=True)# input size = 10, output size = 1, filter size = 10 / 11 = (20-10)/1 + 1 / => (1,11)
        self.conv4 = nn.Conv1d(5, 1, 10, bias=True)
        self.lin_final = nn.Linear(3, 1, bias=True)

        self.activation = activation
        # self.activation_final = nn.LeakyReLU()

    def forward(self, x):
        conv_out = self.conv1(x)
        act_out = self.activation(conv_out)
        mp_out = self.mp(act_out)

        conv2_out = self.activation(self.conv2(mp_out)).squeeze(1) # (1,11) => (11)
        conv3_out = self.activation(self.conv3(conv2_out))
        conv4_out = self.activation(self.conv4(conv3_out))
        lin_out = self.lin_final(conv4_out).squeeze(-1) # (11) => (1)

        return lin_out

#%%

class SpectralCNN(nn.Module):
    def __init__(self, activation=nn.LeakyReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(21, 15, [40, 2], bias=True)
        self.conv2 = nn.Conv1d(15, 10, 10, bias=True)
        self.lin1 = nn.Linear(1470, 80)
        self.lin2 = nn.Linear(80, 1)
        self.activation = activation

    def forward(self, x):
        # return self.lin(x)
        conv1_out = self.activation(self.conv1(x).squeeze(-1))
        conv2_out = self.activation(self.conv2(conv1_out))
        conv2_out_flattened = torch.flatten(conv2_out, start_dim=1)
        lin_out = self.activation(self.lin1(conv2_out_flattened))

        return self.lin2(lin_out).squeeze()

class BeamCNN(nn.Module):
    def __init__(self, activation=nn.LeakyReLU()):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 1, [3, 40, 2], bias=True)
        self.conv2 = nn.Conv2d(1, 4, [5, 10], bias=True)
        self.lin1 = nn.Linear(8820, 400)
        self.lin2 = nn.Linear(400, 1)
        self.lin3 = nn.Linear(50, 1)
        self.activation = activation

    def forward(self, x):
        # return self.lin(x)
        conv1_out = self.activation(self.conv1(x.unsqueeze(1)).squeeze(-1))
        conv2_out = self.activation(self.conv2(conv1_out))
        conv2_out_flattened = torch.flatten(conv2_out, start_dim=1)
        lin_out = self.activation(self.lin1(conv2_out_flattened))
        # lin2_out = self.activation(self.lin2(lin_out))

        return self.lin2(lin_out).squeeze()


class SpectralMLP(nn.Module):
    def __init__(self, activation=nn.LeakyReLU()):
        super().__init__()
        self.lin1 = nn.Linear(8190, 200, bias=True)
        self.lin2 = nn.Linear(200, 100, bias=True)
        self.lin3 = nn.Linear(100, 50)
        self.lin4 = nn.Linear(50, 1)
        self.activation = activation
        self.linear = nn.ModuleList([self.lin1, self.lin2, self.lin3, self.lin4])

    def forward(self, x):
        # return self.lin(x)
        res = x.flatten(start_dim=1)
        for idx, module in enumerate(self.linear):
            res = module(res)
            if idx < len(self.linear) - 1:
                res = self.activation(res)

        return res


class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes, activation=nn.LeakyReLU()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (self.in_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, self.out_channels, self.modes, dtype=torch.cfloat))

    def forward(self, x):
        x_fourier = torch.fft.rfft(x, dim=2)

        out_fourier = torch.zeros(x.shape[0], self.out_channels, x_fourier.shape[2], dtype=torch.cfloat, device=x.device)

        out_fourier[:, :, :self.modes] = torch.einsum('bix,iox->box', (x_fourier[:, :, :self.modes], self.weights))

        return torch.fft.irfft(out_fourier, n=x.shape[2], dim=2)


class FourierLayers(nn.Module):
    def __init__(self, modes, num_layers, activation_func, width=21):
        super().__init__()

        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.activation_func = activation_func

        self.spectral_layers = nn.ModuleList([SpectralConv(width, width, modes) for l in range(num_layers)])

        self.layers = nn.ModuleList([nn.Conv1d(width, width, 800, padding='same') for _ in range(num_layers)])

    def forward(self, x):
        res = x

        for idx, spectral_layer, layer in zip(range(len(self.layers)), self.spectral_layers, self.layers):
            spectral_res = spectral_layer(res)
            # next_t = torch.transpose(next, 1, 2)
            layer_res = layer(res)
            # layer_res = torch.transpose(layer_res, 1, 2)

            if idx < len(self.layers) - 1:
                res = self.activation_func(spectral_res + layer_res)

        return res


class FourierNet(nn.Module):
    def __init__(self, activation_func=nn.LeakyReLU()):
        super().__init__()
        self.fourier_layers = FourierLayers(2500, 2, activation_func=activation_func)
        self.cnn = DeepCNN(activation=activation_func)

    def forward(self, x):
        filtered = self.fourier_layers(x)
        return self.cnn(filtered)
#%%

# mynet = TestNet()
# mycnn = MyCNN()
# xtest, ytest = train_data.dataset.x[0], train_data.dataset.y[0]
# xtest, ytest = torch.tensor(xtest).float(), torch.tensor(ytest).float()
# # xtest = torch.transpose(xtest, 1, 2)
# # print(mynet(xtest))
# # print(mynet(xtest).shape)