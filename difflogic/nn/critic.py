#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np

class GRUCritic(nn.Module):

    def __init__(self,
                 input_dim_state,
                 num_layers=2,
                 hidden_size=64,
                 shuffle_index=False):
        super().__init__()
        self.shuffle_index = shuffle_index
        self.input_dim_state = input_dim_state
        self.gru = torch.nn.GRU(
            input_dim_state,  # + 1,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False)
        #        nonlinearity='relu')
        self.lin = torch.nn.Linear(hidden_size, 1)

    def forward(self, states):
        st = states.reshape((states.shape[0], -1, self.input_dim_state))
        # ac = actions.reshape((states.shape[0], -1, 1))
        f = st  # torch.cat([st,ac], dim=2)

        if self.shuffle_index:
            torch.randperm(f.shape[1])
            for i in range(f.shape[0]):
                f[i, :, :] = f[i, torch.randperm(f.shape[1]), :]

        out, _ = self.gru(f)
        out = out[:, -1, :]

        return self.lin(out).reshape(states.shape[0])


class MLPCritic(nn.Module):

    def __init__(self,
                 input_dim_state,
                 hidden_size=64):
        super().__init__()
        input_dim_state = np.array(input_dim_state).prod()

        self.layers1 = nn.Sequential(
            nn.Linear(input_dim_state, hidden_size),
            nn.ReLU())
        self.layers2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.layers3 = nn.Linear(hidden_size, 1)

#        self.bn = nn.BatchNorm1d(input_dim_state)

    def forward(self, states):
        st = states.reshape((states.shape[0], -1))
#        f = self.bn(st)
        f = self.layers1(st)
        f = self.layers2(f)
        f = self.layers3(f)

        return f.reshape(states.shape[0])


class NLMMLPCritic(nn.Module):

    def __init__(self,
                 input_dim_state,
                 nlm,
                 nlm_breadth=3,
                 hidden_size=64,
                 feature_axis=None):
        super().__init__()

        input_dim_state = input_dim_state[0]
        self.nlm = nlm
        self.nlm_breadth = nlm_breadth
        self.feature_axis = feature_axis
        output_size = 0
        for i in range(len(self.nlm.output_dims)):
            output_size += self.nlm.output_dims[i] * (input_dim_state ** i)

        self.layers2 = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU())
        self.layers3 = nn.Linear(hidden_size, 1)

    def forward(self, states):
        inp = [None for i in range(self.nlm_breadth)]
        inp[2] = states

        f = self.nlm(inp, depth=None)
        if self.feature_axis is not None:
            f = f[self.feature_axis]
        for i in range(len(f)):
            f[i] = f[i].reshape(f[i].shape[0], -1)

        f = torch.cat(f, dim=1)

        f = self.layers2(f)
        f = self.layers3(f)

        return f.reshape(states.shape[0])


class ConvCritic(nn.Module):
    def __init__(self,
                 nbobject,
                 input_channel,
                 hidden_size=64):
        super().__init__()

        nbobject = np.array(nbobject)[:2].prod()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            )
        self.drop_out = nn.Dropout()

        self.fc1 = nn.Sequential(
            nn.Linear(64 * nbobject, hidden_size),
            nn.ReLU())
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, states):
        f = states.transpose(1, 3)
        f = self.layer1(f)
        f = self.layer2(f)

        f = f.reshape(f.size(0), -1)
        # f = self.drop_out(f)
        f = self.fc1(f)
        f = self.fc2(f)

        return f.reshape(states.shape[0])


class MixedGRUCritic(nn.Module):
    def __init__(self,
                 input_channel,
                 hidden_size=64,
                 num_layers=1):
        super().__init__()

        if type(input_channel) is not list:
            self.input_channel = [0, 0, input_channel]
        else:
            self.input_channel = input_channel

        if len(self.input_channel) > 2 and self.input_channel[2] > 0:
            self.layer2Dto1D = nn.ModuleList([torch.nn.GRU(
                self.input_channel[2],
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=False)] * 2)

        inter_size = hidden_size * 2 + 6 * (self.input_channel[2]) + self.input_channel[1]
        self.layer1DtoScalar = torch.nn.GRU(
            inter_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False)

        self.fc1 = nn.Sequential(
            nn.Linear(inter_size * 3 + hidden_size, hidden_size),
            nn.ReLU())
        self.fc2 = nn.Linear(hidden_size, 1)

    def flatten_parameters(self):
        self.layer2Dto1D[0].flatten_parameters()
        self.layer2Dto1D[1].flatten_parameters()
        self.layer1DtoScalar.flatten_parameters()

    def forward(self, allstates):
        states = allstates[1]
        d1 = []
        d1.append(states.max(dim=1)[0])
        d1.append(states.mean(dim=1))
        d1.append(states.min(dim=1)[0])

        d1.append(states.max(dim=2)[0])
        d1.append(states.mean(dim=2))
        d1.append(states.min(dim=2)[0])

        sd1 = []
        sd2 = []
        for i in range(states.shape[1]):
            sd1.append(self.layer2Dto1D[0](states[:, i, :, :])[0][:, -1, :])
            sd2.append(self.layer2Dto1D[1](states[:, :, i, :])[0][:, -1, :])
        d1.append(torch.stack(sd1, dim=1))
        d1.append(torch.stack(sd2, dim=1))
        f = torch.cat(d1, dim=2)

        if allstates[0] is not None:
            f = torch.cat([allstates[0], f], -1)

        d1 = []
        d1.append(f.max(dim=1)[0])
        d1.append(f.mean(dim=1))
        d1.append(f.min(dim=1)[0])
        d1.append(self.layer1DtoScalar(f)[0][:, -1, :])
        f = torch.cat(d1, dim=1)

        f = self.fc1(f)
        f = self.fc2(f)

        return f.reshape(states.shape[0])


class MLPCriticQ(nn.Module):

    def __init__(self,
                 input_dim_state,
                 n_action_func,
                 hidden_size=64):
        super().__init__()
        n = n_action_func(input_dim_state)
        input_dim_state = np.array(input_dim_state).prod()

        self.layers1 = nn.Sequential(
            nn.Linear(input_dim_state, hidden_size),
            nn.ReLU())
        self.layers2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.layers3 = nn.Linear(hidden_size, n)

        self.input_dim_state = input_dim_state
#        self.dropout = nn.Dropout()

    def forward(self, states):
        f = states.reshape((states.shape[0], -1))

        f = self.layers1(f)
#        f = self.dropout(f)
        f = self.layers2(f)
#        f = self.dropout(f)
        f = self.layers3(f)

        return f


class NLMMLPCriticQ(nn.Module):

    def __init__(self,
                 input_dim_state,
                 nlm,
                 n_action_func,
                 nlm_breadth=3,
                 feature_axis=None):
        super().__init__()

        self.nlm = nlm
        self.nlm_breadth = nlm_breadth
        self.feature_axis = feature_axis

        self.lastlayer = nn.Linear(self.nlm.output_dims[2], n_action_func(input_dim_state))

    def forward(self, states):
        inp = [None for i in range(self.nlm_breadth)]
        inp[2] = states

        if self.feature_axis is None:
            f = self.nlm(inp, depth=None)[2]
        else:
            f = self.nlm(inp, depth=None)[self.feature_axis][2]
        f = self.lastlayer(f)
        return f.mean((1,2))


class InvariantNObject(nn.Module):
    def __init__(self, critic_class, range_dim, *args):
        super().__init__()

        self.critics = {}
        for rd in range_dim:
            self.critics[rd[0]] = critic_class(rd, **args[0])
            self.add_module(str(rd[0]), self.critics[rd[0]])

    def forward(self, input):
        key = input.shape[1]
        return self.critics[key](input)


class ConvReduceCritic(nn.Module):
    def __init__(self,
                 nbobject,
                 hidden_size=64):
        super().__init__()

        n = nbobject[0]
        input_channel = nbobject[-1]

        two_layer = False
        if n <= 6:
            kernel1 = 5
        elif n <= 8:
            kernel1 = 6
        elif n <= 10:
            kernel1 = 8
        elif n <= 14:
            kernel1 = 6
            kernel2 = 3
            two_layer = True
        elif n <= 18:
            kernel1 = 8
            kernel2 = 3
            two_layer = True
        elif n <= 26:
            kernel1 = 8
            kernel2 = 4
            two_layer = True

        stride = kernel1 // 2
        l = [
            nn.Conv2d(input_channel, 32, kernel_size=kernel1, stride=stride, padding=0),
            nn.ReLU()
        ]

        if two_layer:
            stride = kernel2 // 2
            l.append(nn.Conv2d(32, 64, kernel_size=kernel2, stride=stride, padding=0))
            l.append(nn.ReLU())

        l.append(nn.Flatten())

        self.layer1 = nn.Sequential(*l)
        n_flatten = self.layer1(torch.zeros(size=nbobject).unsqueeze(0).transpose(1, 3)).shape[1]

        self.linear = nn.Linear(n_flatten, 1)

    def forward(self, states):

        f = states.transpose(1, 3)
        f = self.layer1(f)
        f = self.linear(f)

        return f.reshape(states.shape[0])


class ConvReduceCriticQ(nn.Module):
    def __init__(self,
                 nbobject,
                 n_action_func,
                 hidden_size=64):
        super().__init__()

        n = nbobject[0]
        n_ac = n_action_func(nbobject)
        input_channel = nbobject[-1]

        two_layer = False
        if n <= 6:
            kernel1 = 5
        elif n <= 8:
            kernel1 = 6
        elif n <= 10:
            kernel1 = 8
        elif n <= 14:
            kernel1 = 6
            kernel2 = 3
            two_layer = True
        elif n <= 18:
            kernel1 = 8
            kernel2 = 3
            two_layer = True
        elif n <= 26:
            kernel1 = 8
            kernel2 = 4
            two_layer = True

        stride = kernel1 // 2
        l = [
            nn.Conv2d(input_channel, 32, kernel_size=kernel1, stride=stride, padding=0),
            nn.ReLU()
        ]

#        print(kernel1, nbobject, nn.Sequential(*l)(torch.zeros(size=nbobject).unsqueeze(0).transpose(1, 3)).shape)
        if two_layer:
            stride = kernel2 // 2
            l.append(nn.Conv2d(32, 64, kernel_size=kernel2, stride=stride, padding=0))
            l.append(nn.ReLU())

#        if two_layer:
#            print(kernel2, nbobject, nn.Sequential(*l)(torch.zeros(size=nbobject).unsqueeze(0).transpose(1, 3)).shape)

        l.append(nn.Flatten())

        self.layer1 = nn.Sequential(*l)

#        print(self.layer1(torch.zeros(size=nbobject).unsqueeze(0).transpose(1, 3)).shape)
#        breakpoint()
        n_flatten = self.layer1(torch.zeros(size=nbobject).unsqueeze(0).transpose(1, 3)).shape[1]

        self.linear = nn.Linear(n_flatten, n_ac)
#        self.linear = nn.Sequential(nn.Linear(n_flatten, 128), nn.ReLU(), nn.Linear(128, n))

    def forward(self, states):

        f = states.transpose(1, 3)
        f = self.layer1(f)
        f = self.linear(f)

        return f


class ActionSelector(nn.Module):
    def __init__(self,
                 input_channel,
                 hidden_size=32,
                 num_layers=1):
        super().__init__()

        self.mixed_part = torch.nn.GRU(
           input_channel,
           hidden_size,
           num_layers,
           batch_first=True,
           bidirectional=True)

        inter_size = hidden_size * 2 + input_channel
        self.fc1 = nn.Sequential(
            nn.Linear(inter_size, hidden_size),
            nn.ReLU())
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, states):
        states = states.reshape(states.shape[0], -1, states.shape[-1])

        f = self.mixed_part(states)[0]
        f = torch.cat([states, f], -1)

        f = self.fc1(f)
        f = self.fc2(f)

        return f[..., 0]

