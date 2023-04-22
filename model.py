import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess import load_data


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        # FIXME: 64 * 6 * 6 may not be correct
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.pool1(self.relu1(self.conv1(x)))
        # print(x.shape)
        x = self.pool2(self.relu2(self.conv2(x)))
        # flatten the dimension except for batch
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# my model
import torch
from torch import nn, Tensor

# FIXME resnet structure, data augmentation

from torchvision.models.resnet import ResNet, Bottleneck

# my_nn = ResNet(layers=[2, 2, 2, 2], num_classes=10, block=Bottleneck)

class MyNet(nn.Module):
    # without dilate
    def __init__(self, layers: list[int], num_classes: int = 10):
        super().__init__()
        self.inplanes = 64
        # norm layer FIXME change the features
        self._norm_layer = nn.BatchNorm2d

        # input layers
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # bottleneck layers
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_classes)
        # self.fc = nn.Linear(1000, num_classes)
        # probability outputs
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                self._norm_layer(planes * Bottleneck.expansion),
            )

        layers = [Bottleneck(
            self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer
        )]

        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, norm_layer=self._norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    """
    Epoch 1/10, train loss: 0.0019, train accuracy: 75.2271
Epoch 2/10, train loss: 0.0014, train accuracy: 79.9339
Epoch 3/10, train loss: 0.0012, train accuracy: 83.9102
Epoch 4/10, train loss: 0.0010, train accuracy: 85.9339
Epoch 5/10, train loss: 0.0008, train accuracy: 87.7864
Epoch 6/10, train loss: 0.0006, train accuracy: 89.2508
Epoch 7/10, train loss: 0.0005, train accuracy: 89.9271
Epoch 8/10, train loss: 0.0003, train accuracy: 89.9102
Epoch 9/10, train loss: 0.0003, train accuracy: 91.9576
Epoch 10/10, train loss: 0.0002, train accuracy: 93.6593
Finished Training

Process finished with exit code 0
"""