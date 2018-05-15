import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import norm_col_init, weights_init

class locNet(torch.nn.Module):
    def __init__(self, num_inputs = 3, num_outputs = 6):
        super(locNet, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.ff0 = nn.Linear(9600, 1200)
        self.ff1 = nn.Linear(1200, 96)

        self.lstm = nn.LSTM(288, 72)

        self.f0 = nn.Linear(6, 96)
        self.f1 = nn.Linear(6, 96)

        self.fc1 = nn.Linear(72, 36)
        self.fc2 = nn.Linear(36, 12)
        self.fc3 = nn.Linear(12, num_outputs)

        self.apply(weights_init)
        self.ff0.weight.data = norm_col_init(
            self.ff0.weight.data, 1.0)
        self.ff0.bias.data.fill_(0)
        
        self.ff1.weight.data = norm_col_init(
            self.ff1.weight.data, 1.0)
        self.ff1.bias.data.fill_(0)

        self.f0.weight.data = norm_col_init(
            self.f0.weight.data, 1.0)
        self.f0.bias.data.fill_(0)

        self.f1.weight.data = norm_col_init(
            self.f1.weight.data, 1.0)
        self.f1.bias.data.fill_(0)
        
        self.fc1.weight.data = norm_col_init(
            self.fc1.weight.data, 1.0)
        self.fc1.bias.data.fill_(0)

        self.fc2.weight.data = norm_col_init(
            self.fc2.weight.data, 1.0)
        self.fc2.bias.data.fill_(0)

        self.fc3.weight.data = norm_col_init(
            self.fc3.weight.data, 1.0)
        self.fc3.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        pic, force, action, (hx, cx) = inputs
        x = F.relu(F.max_pool2d(self.conv1(pic), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
        x = x.view(x.size(0), -1).unsqueeze(1) # (10,1,len)
        x = self.ff0(x)
        x = self.ff1(x)

        force = self.f0(force)
        action = self.f1(action)
        #x =  torch.cat([x, force, action],1).view(,1,-1) # concatenate
        x =  torch.cat([x, force, action],2)
        hx, cx = self.lstm(x, (hx, cx))

        x = F.relu(self.fc1(hx))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x, (hx, cx)




class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels, self.kernel_size, 1,
                              self.padding)

    def forward(self, input, h, c):

        combined = torch.cat((input, h), dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, A.size()[1] / self.num_features, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = ConvLSTMCell.init_hidden(bsize, self.hidden_channels[i], (height, width))
                    internal_state.append((h, c))
                # do forward
                name = 'cell{}'.format(i)
                (h, c) = internal_state[i]

                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)