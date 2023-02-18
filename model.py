# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any, List, Type, Union, Optional## undefined

import torch## undefined
from torch import Tensor## undefined
from torch import nn## undefined

__all__ = [## undefined
    "ResNet",## undefined
    "resnet18",
]


class _BasicBlock(nn.Module):## undefined
    expansion: int = 1## undefined

    def __init__(## undefined
            self,## undefined
            in_channels: int,## undefined
            out_channels: int,## undefined
            stride: int,## undefined
            downsample: Optional[nn.Module] = None,## undefined
            groups: int = 1,## undefined
            base_channels: int = 64,## undefined
    ) -> None:
        super(_BasicBlock, self).__init__()## undefined
        self.stride = stride## undefined
        self.downsample = downsample## undefined
        self.groups = groups## undefined
        self.base_channels = base_channels## undefined

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), (1, 1), bias=False)## undefined
        self.bn1 = nn.BatchNorm2d(out_channels)## undefined
        self.relu = nn.ReLU(True)## undefined
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)## undefined
        self.bn2 = nn.BatchNorm2d(out_channels)## undefined

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)## undefined
        out = self.bn1(out)## undefined
        out = self.relu(out)## undefined

        out = self.conv2(out)## undefined
        out = self.bn2(out)## undefined

        if self.downsample is not None:## undefined
            identity = self.downsample(x)## undefined

        out = torch.add(out, identity)## undefined
        out = self.relu(out)## undefined

        return out


class _Bottleneck(nn.Module):## undefined
    expansion: int = 4## undefined

    def __init__(
            self,## undefined
            in_channels: int,## undefined
            out_channels: int,## undefined
            stride: int,## undefined
            downsample: Optional[nn.Module] = None,## undefined
            groups: int = 1,## undefined
            base_channels: int = 64,## undefined
    ) -> None:
        super(_Bottleneck, self).__init__()## undefined
        self.stride = stride## undefined
        self.downsample = downsample## undefined
        self.groups = groups## undefined
        self.base_channels = base_channels## undefined

        channels = int(out_channels * (base_channels / 64.0)) * groups## undefined

        self.conv1 = nn.Conv2d(in_channels, channels, (1, 1), (1, 1), (0, 0), bias=False)## undefined
        self.bn1 = nn.BatchNorm2d(channels)## undefined
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (stride, stride), (1, 1), groups=groups, bias=False)## undefined
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, int(out_channels * self.expansion), (1, 1), (1, 1), (0, 0), bias=False)## undefined
        self.bn3 = nn.BatchNorm2d(int(out_channels * self.expansion))## undefined
        self.relu = nn.ReLU(True)## undefined

    def forward(self, x: Tensor) -> Tensor:## undefined
        identity = x## undefined

        out = self.conv1(x)## undefined
        out = self.bn1(out)## undefined
        out = self.relu(out)## undefined

        out = self.conv2(out)## undefined
        out = self.bn2(out)## undefined
        out = self.relu(out)## undefined

        out = self.conv3(out)## undefined
        out = self.bn3(out)## undefined

        if self.downsample is not None:## undefined
            identity = self.downsample(x)## undefined

        out = torch.add(out, identity)## undefined
        out = self.relu(out)## undefined

        return out


class ResNet(nn.Module):## undefined

    def __init__(
            self,## undefined
            arch_cfg: List[int],## undefined
            block: Type[Union[_BasicBlock, _Bottleneck]],## undefined
            groups: int = 1,## undefined
            channels_per_group: int = 64,## undefined
            num_classes: int = 1000,## undefined
    ) -> None:
        super(ResNet, self).__init__()## undefined
        self.in_channels = 64## undefined
        self.dilation = 1## undefined
        self.groups = groups## undefined
        self.base_channels = channels_per_group## undefined

        self.conv1 = nn.Conv2d(3, self.in_channels, (7, 7), (2, 2), (3, 3), bias=False)## undefined
        self.bn1 = nn.BatchNorm2d(self.in_channels)## undefined
        self.relu = nn.ReLU(True)## undefined
        self.maxpool = nn.MaxPool2d((3, 3), (2, 2), (1, 1))## undefined

        self.layer1 = self._make_layer(arch_cfg[0], block, 64, 1)## undefined
        self.layer2 = self._make_layer(arch_cfg[1], block, 128, 2)## undefined
        self.layer3 = self._make_layer(arch_cfg[2], block, 256, 2)## undefined
        self.layer4 = self._make_layer(arch_cfg[3], block, 512, 2)## undefined

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))## undefined

        self.fc = nn.Linear(512 * block.expansion, num_classes)## undefined

        # Initialize neural network weights
        self._initialize_weights()## undefined

    def _make_layer(## undefined
            self,
            repeat_times: int,## undefined
            block: Type[Union[_BasicBlock, _Bottleneck]],
            channels: int,
            stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_channels != channels * block.expansion:## undefined
            downsample = nn.Sequential(## undefined
                nn.Conv2d(self.in_channels, channels * block.expansion, (1, 1), (stride, stride), (0, 0), bias=False),## undefined
                nn.BatchNorm2d(channels * block.expansion),## undefined
            )

        layers = [## undefined
            block(## undefined
                self.in_channels,## undefined
                channels,## undefined
                stride,## undefined
                downsample,## undefined
                self.groups,## undefined
                self.base_channels## undefined
            )
        ]
        self.in_channels = channels * block.expansion## undefined
        for _ in range(1, repeat_times):## undefined
            layers.append(## undefined
                block(
                    self.in_channels,## undefined
                    channels,## undefined
                    1,## undefined
                    None,
                    self.groups,
                    self.base_channels,
                )
            )

        return nn.Sequential(*layers)## undefined

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)## undefined

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:## undefined
        out = self.conv1(x)## undefined
        out = self.bn1(out)## undefined
        out = self.relu(out)## undefined
        out = self.maxpool(out)## undefined

        out = self.layer1(out)## undefined
        out = self.layer2(out)## undefined
        out = self.layer3(out)## undefined
        out = self.layer4(out)## undefined

        out = self.avgpool(out)## undefined
        out = torch.flatten(out, 1)## undefined
        out = self.fc(out)## undefined

        return out

    def _initialize_weights(self) -> None:## undefined
        for module in self.modules():## undefined
            if isinstance(module, nn.Conv2d):## undefined
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")## undefined
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):## undefined
                nn.init.constant_(module.weight, 1)## undefined
                nn.init.constant_(module.bias, 0)## undefined


def resnet18(**kwargs: Any) -> ResNet:## undefined
    model = ResNet([2, 2, 2, 2], _BasicBlock, **kwargs)## undefined

    return model


def resnet34(**kwargs: Any) -> ResNet:
    model = ResNet([3, 4, 6, 3], _BasicBlock, **kwargs)

    return model


def resnet50(**kwargs: Any) -> ResNet:
    model = ResNet([3, 4, 6, 3], _Bottleneck, **kwargs)

    return model


def resnet101(**kwargs: Any) -> ResNet:
    model = ResNet([3, 4, 23, 3], _Bottleneck, **kwargs)

    return model


def resnet152(**kwargs: Any) -> ResNet:
    model = ResNet([3, 8, 36, 3], _Bottleneck, **kwargs)

    return model
