from typing import Tuple

import numpy as np

import torch
from torch import nn
from torchvision.models import resnet18, ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock


class ResNetFeatures(ResNet):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        # Override init
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class HeatMapGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_rate, scale_factor=2, kernel_size=3):
        super(HeatMapGenerator, self).__init__()


        ch_size = in_channels
        self.layers = []

        pad = int((kernel_size-1) / 2)

        number_of_upsacles = np.log(upscale_rate) / np.log(scale_factor)
        assert number_of_upsacles.is_integer(), f'number of upsalce is not integer ({number_of_upsacles})' \
                                                f'\n trying to upscale by {upscale_rate} with factor {scale_factor}'

        for i in range(int(number_of_upsacles)):
            ch_next = ch_size // 2
            l = nn.Sequential(
                nn.Conv2d(
                    in_channels=ch_size,
                    out_channels=ch_next,
                    kernel_size=kernel_size,
                    padding=pad
                ),
                nn.Upsample(scale_factor=scale_factor),
                # nn.ConvTranspose2d(
                #     in_channels=ch_size,
                #     out_channels=ch_next,
                #     kernel_size=3,
                #     stride=2
                # ),
                nn.BatchNorm2d(num_features=ch_next),
                nn.ReLU()

            )
            self.layers.append(l)
            ch_size = ch_next

        self.out_conv = nn.Conv2d(
            in_channels=ch_size,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        x = fmap
        for l in self.layers:
            x = l(x)

        x = self.out_conv(x)
        return x

def bottleneck_resnet(block_sizes=(2, 2, 2, 2), base_channels=32) -> (nn.Module, Tuple, int):
    _resnet = ResNetFeatures(
        block=Bottleneck,
        layers=block_sizes,
        width_per_group=base_channels
    )

    fmap = _resnet(torch.ones(size=(2,3, 640,928), dtype=torch.float32))

    shrink_hw = (640 / fmap.shape[2], 928 / fmap.shape[3])

    return _resnet, shrink_hw, fmap.shape[1]



if __name__ == '__main__':
    x = torch.ones(size=(2,3, 640,928), dtype=torch.float32)
    model = ResNetFeatures(
        block=Bottleneck,
        layers=[2,2,2,2],
    )

    hmap = HeatMapGenerator(in_channels=2048, out_channels=1)

    y = model(x)

    x_ = hmap(y)
    print('.')