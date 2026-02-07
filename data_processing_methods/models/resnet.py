import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# BasicBlock for ResNet-18 and ResNet-34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Bottleneck for ResNet-50, ResNet-101, and ResNet-152
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet class
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def load_pretrained_weights(local_model, pretrained_model):
    pretrained_dict = pretrained_model.state_dict()
    model_dict = local_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc')}
    model_dict.update(pretrained_dict)
    local_model.load_state_dict(model_dict)


# Functions to create ResNet-18, ResNet-50, and ResNet-101
def ResNet18(num_classes=5, isPretrained=True):
    local_model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    pretrained_resnet = models.resnet18(pretrained=isPretrained)
    load_pretrained_weights(local_model, pretrained_resnet)
    local_model.fc = nn.Linear(local_model.fc.in_features, num_classes)
    return local_model


def ResNet50(num_classes=5, isPretrained=True):
    local_model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    pretrained_resnet = models.resnet50(pretrained=isPretrained)
    load_pretrained_weights(local_model, pretrained_resnet)
    local_model.fc = nn.Linear(local_model.fc.in_features, num_classes)
    return local_model


def ResNet101(num_classes=5, isPretrained=True):
    local_model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    pretrained_resnet = models.resnet101(pretrained=isPretrained)
    load_pretrained_weights(local_model, pretrained_resnet)
    local_model.fc = nn.Linear(local_model.fc.in_features, num_classes)
    return local_model


class ResNeXtForClassification(nn.Module):
    def __init__(self, num_classes=5, is_pretrained=True):
        super(ResNeXtForClassification, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=is_pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ResGANet
class GatedAttention(nn.Module):
    def __init__(self, in_channels, gate_channels) -> None:
        super(GatedAttention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels, gate_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(gate_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, gate_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(gate_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(gate_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi


class ResGANet(ResNet):
    def __init__(self, block, layers, num_classes=100):
        super(ResGANet, self).__init__(block, layers, num_classes=num_classes)
        self.gate1 = GatedAttention(64, 32)
        self.gate2 = GatedAttention(256, 128)
        self.gate3 = GatedAttention(512, 256)
        self.gate4 = GatedAttention(1024, 512)
        self.relu = nn.ReLU()

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x1 = self.gate1(x1, x1)

        x2 = self.layer2(x1)
        x2 = self.gate2(x2, x2)

        x3 = self.layer3(x2)
        x3 = self.gate3(x3, x3)

        x4 = self.layer4(x3)
        x4 = self.gate4(x4, x4)

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResGANet18(num_classes=5):
    return ResGANet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResGANet34(num_classes=5):
    return ResGANet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResGANet50(num_classes=5):
    return ResGANet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


if __name__ == "__main__":
    try:
        model = ResGANet18(num_classes=5)
        print("Model load successful.")
    except Exception as e:
        print(f"Model load error: {e}")