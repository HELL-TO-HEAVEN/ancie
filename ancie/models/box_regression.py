from typing import Tuple

import numpy as np

import torch
from torch import nn
from torchvision.models import resnet18, ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock


def _get_block(in_chans, out_chans, num_convs, kernel_size, act_fn=nn.ReLU, first_stride=1):
    block = [
        nn.Conv2d(
            in_channels=in_chans,
            out_channels=out_chans,
            kernel_size=kernel_size,
            stride=first_stride,
            padding=int((kernel_size-1) / 2)
        ),
        nn.BatchNorm2d(out_chans),
        act_fn()
    ]

    for conv_n in range(num_convs):
        block.append(
            nn.Conv2d(
                in_channels=out_chans,
                out_channels=out_chans,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size-1) / 2)
            )
        )

        block.append(
            nn.BatchNorm2d(out_chans)
        )

        block.append(
            act_fn()
        )

    return nn.Sequential(*block)


class BoxRegressor(nn.Module):

    def __init__(
            self,
            in_channels,
            downscale_ratio,
            base_channels=32,
            act_fn=nn.ReLU,
            box_parametrization_dim=4
    ):
        super(BoxRegressor, self).__init__()

        layers = [
            _get_block(
                in_chans=in_channels,
                out_chans=base_channels,
                first_stride=2,
                num_convs=3,
                kernel_size=3
            )
        ]

        down_scales = np.log2(downscale_ratio)
        assert down_scales.is_integer(), f'downscale_ratio must be power of 2 (got {downscale_ratio})'
        down_scales = int(down_scales)

        for ds_step in range(down_scales - 1):
            ch = base_channels * (2 ** (ds_step))
            layers.append(
                _get_block(
                    in_chans=ch,
                    out_chans=2*ch,
                    num_convs=3,
                    kernel_size=3,
                    first_stride=2,
                    act_fn=act_fn
                )
            )

        box_reg_feature_dim = base_channels * (2 ** (down_scales-1))

        layers += [
            _get_block(
                in_chans=box_reg_feature_dim,
                out_chans=box_reg_feature_dim,
                num_convs=0,
                kernel_size=3,
                first_stride=1,
                act_fn=act_fn,
            )
            for _ in range(2)
        ]

        self.layers = nn.ModuleList(layers)

        self.box_predictor = nn.Conv2d(
            in_channels=box_reg_feature_dim,
            out_channels=box_parametrization_dim,
            kernel_size=1
        )

        self.init_weights()

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)

        return self.box_predictor(x)

if __name__ == '__main__':
    x = torch.ones((2, 3, 1088, 1024))
    reg = BoxRegressor(
        in_channels=3,
        downscale_ratio=32
    )
    y = reg(x)

    print('/')