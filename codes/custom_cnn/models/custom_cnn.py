"""
Custom CNN models for CIFAR-10 classification.

Includes:
- CustomCNN: Base CNN with Conv, BN, ReLU, MaxPool, Dropout, FC
- CustomResNet: CNN with residual connections
- Different width/depth configurations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()
    return parameters_n


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and optional dimension matching."""

    def __init__(self, in_channels, out_channels, stride=1, use_bn=True,
                 use_dropout=False, dropout_rate=0.3):
        super().__init__()
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=not use_bn)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=not use_bn)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout2d(dropout_rate)

        # shortcut to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut_layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                         stride=stride, bias=not use_bn)]
            if use_bn:
                shortcut_layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = F.relu(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out += shortcut
        out = F.relu(out)
        return out


class CustomCNN(nn.Module):
    """Custom CNN for CIFAR-10.

    Architecture: Conv -> BN -> ReLU -> MaxPool -> ... -> FC -> Dropout -> FC

    Args:
        filters: list of filter counts per stage
        num_blocks: number of conv layers per stage
        fc_size: size of the fully-connected hidden layer
        use_bn: whether to use BatchNorm
        use_dropout: whether to use Dropout
        dropout_rate: dropout probability
        activation: activation function ('relu', 'leaky_relu', 'gelu')
    """

    def __init__(self, filters=(64, 128, 256), num_blocks=(2, 2, 2),
                 fc_size=256, num_classes=10, use_bn=True, use_dropout=True,
                 dropout_rate=0.3, activation='relu'):
        super().__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        # activation function
        if activation == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'gelu':
            self.act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # convolutional stages
        stages = []
        in_ch = 3
        feature_size = 32  # input size

        for stage_idx, (filt, n_blocks) in enumerate(zip(filters, num_blocks)):
            for block_idx in range(n_blocks):
                out_ch = filt
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1,
                                 bias=not use_bn)
                stages.append(conv)
                if use_bn:
                    stages.append(nn.BatchNorm2d(out_ch))
                stages.append(self._get_activation())
                in_ch = out_ch

            # max pooling at end of each stage (except last)
            if stage_idx < len(filters) - 1:
                stages.append(nn.MaxPool2d(2, 2))
                feature_size //= 2

        # final pooling: adaptive average pool to 1x1
        self.features = nn.Sequential(*stages)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # classifier
        classifier_layers = []
        if use_dropout:
            classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.extend([
            nn.Linear(filters[-1], fc_size),
            self._get_activation(),
        ])
        if use_dropout:
            classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(fc_size, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        self._init_weights()

    def _get_activation(self):
        if isinstance(self.act_fn, nn.ReLU):
            return nn.ReLU(inplace=True)
        elif isinstance(self.act_fn, nn.LeakyReLU):
            return nn.LeakyReLU(0.1, inplace=True)
        elif isinstance(self.act_fn, nn.GELU):
            return nn.GELU()
        return nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)


class CustomResNet(nn.Module):
    """Custom ResNet-like model with residual connections for CIFAR-10.

    Uses residual blocks with skip connections.

    Args:
        filters: list of filter counts per stage
        num_blocks: number of residual blocks per stage
        fc_size: size of the fully-connected hidden layer
        use_bn: whether to use BatchNorm
        use_dropout: whether to use Dropout
        dropout_rate: dropout probability
    """

    def __init__(self, filters=(64, 128, 256), num_blocks=(2, 2, 2),
                 fc_size=256, num_classes=10, use_bn=True, use_dropout=True,
                 dropout_rate=0.3):
        super().__init__()
        self.use_bn = use_bn

        # initial conv
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=3, stride=1,
                               padding=1, bias=not use_bn)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(filters[0])

        # residual stages
        self.stages = nn.ModuleList()
        in_ch = filters[0]
        for stage_idx, (filt, n_blocks) in enumerate(zip(filters, num_blocks)):
            stage = []
            for block_idx in range(n_blocks):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                stage.append(ResidualBlock(
                    in_ch, filt, stride=stride,
                    use_bn=use_bn, use_dropout=use_dropout,
                    dropout_rate=dropout_rate
                ))
                in_ch = filt
            self.stages.append(nn.Sequential(*stage))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # classifier
        classifier_layers = []
        if use_dropout:
            classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(filters[-1], fc_size))
        classifier_layers.append(nn.ReLU(inplace=True))
        if use_dropout:
            classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(fc_size, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)


# Predefined model configurations
def custom_cnn_small(**kwargs):
    """Small CNN: fewer filters"""
    return CustomCNN(filters=(32, 64, 128), num_blocks=(2, 2, 2),
                     fc_size=128, **kwargs)


def custom_cnn_medium(**kwargs):
    """Medium CNN: balanced"""
    return CustomCNN(filters=(64, 128, 256), num_blocks=(2, 2, 2),
                     fc_size=256, **kwargs)


def custom_cnn_large(**kwargs):
    """Large CNN: more filters and deeper"""
    return CustomCNN(filters=(64, 128, 256, 512), num_blocks=(2, 2, 2, 2),
                     fc_size=512, **kwargs)


def custom_resnet_small(**kwargs):
    """Small ResNet"""
    return CustomResNet(filters=(32, 64, 128), num_blocks=(2, 2, 2),
                        fc_size=128, **kwargs)


def custom_resnet_medium(**kwargs):
    """Medium ResNet"""
    return CustomResNet(filters=(64, 128, 256), num_blocks=(2, 2, 2),
                        fc_size=256, **kwargs)


def custom_resnet_large(**kwargs):
    """Large ResNet: deeper"""
    return CustomResNet(filters=(64, 128, 256, 512), num_blocks=(2, 2, 2, 2),
                        fc_size=512, **kwargs)


if __name__ == '__main__':
    print("Model Parameter Counts:")
    for name, model_fn in [
        ('CustomCNN-Small', custom_cnn_small),
        ('CustomCNN-Medium', custom_cnn_medium),
        ('CustomCNN-Large', custom_cnn_large),
        ('CustomResNet-Small', custom_resnet_small),
        ('CustomResNet-Medium', custom_resnet_medium),
        ('CustomResNet-Large', custom_resnet_large),
    ]:
        model = model_fn()
        print(f"  {name}: {get_number_of_parameters(model):,} params")
