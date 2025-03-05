from functools import partial
from math import floor

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            num_layers: int,
            activation: nn.Module = nn.ReLU,
            layer_norm: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.layers = nn.Sequential()
        current_in_features = in_features
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(current_in_features, hidden_features, bias=bias))
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_features))
            self.layers.append(activation())
            current_in_features = hidden_features
        self.layers.append(nn.Linear(current_in_features, out_features, bias=bias))

    def forward(self, x):
        return self.layers(x)
    

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        img_shape = (4, 86, 86)
        num_channels = img_shape[0]
        conv_out_channels = 16
        conv_kernel = 5
        conv_stride = 2
        conv_padding = 2

        self.img1_conv1 = nn.Conv2d(num_channels, conv_out_channels, conv_kernel, stride=conv_stride, padding=conv_padding)
        self.img1_conv2 = nn.Conv2d(conv_out_channels, conv_out_channels, conv_kernel, stride=conv_stride, padding=conv_padding)
        self.img2_conv1 = nn.Conv2d(num_channels, conv_out_channels, conv_kernel, stride=conv_stride, padding=conv_padding)
        self.img2_conv2 = nn.Conv2d(conv_out_channels, conv_out_channels, conv_kernel, stride=conv_stride, padding=conv_padding)

        pool_kernel = 5
        pool_stride = 2
        pool_padding = 2
        self.maxpool = nn.MaxPool2d(pool_kernel, stride=pool_stride, padding=pool_padding)
        self.activation = nn.ReLU()

        def compute_shape(size: int, kernel: int, stride: int, padding: int):
            return floor((size + 2 * padding - (kernel - 1) - 1) / stride + 1)
        f_conv = partial(compute_shape, kernel=conv_kernel, stride=conv_stride, padding=conv_padding)
        f_pool = partial(compute_shape, kernel=pool_kernel, stride=pool_stride, padding=pool_padding)
        c = conv_out_channels
        h = f_pool(f_conv(f_pool(f_conv(img_shape[-1]))))
        w = h
        self.out_features = 2 * c * h * w

    def forward(self, img):
        img1 = img[:, 0::2]
        img1 = self.activation(self.maxpool(self.img1_conv1(img1)))
        img1 = self.activation(self.maxpool(self.img1_conv2(img1)))
        img1_feature = torch.flatten(img1, start_dim=1)

        img2 = img[:, 1::2]
        img2 = self.activation(self.maxpool(self.img2_conv1(img2)))
        img2 = self.activation(self.maxpool(self.img2_conv2(img2)))
        img2_feature = torch.flatten(img2, start_dim=1)

        return torch.cat((img1_feature, img2_feature), dim=-1)
