# This code is from HandOccNet (https://github.com/mks0601/Hand4Whole_RELEASE/blob/main/common/nets/resnet.py)
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type):
	
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
		       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
		       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
		       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
		       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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

    def init_weights(self):
        import torchvision.models as models

        if self.name == 'resnet18':
            org_resnet = models.resnet18(pretrained=True)
        elif self.name == 'resnet34':
            org_resnet = models.resnet34(pretrained=True)
        elif self.name == 'resnet50':
            org_resnet = models.resnet50(pretrained=True)
        elif self.name == 'resnet101':
            org_resnet = models.resnet101(pretrained=True)
        elif self.name == 'resnet152':
            org_resnet = models.resnet152(pretrained=True)
        else:
            raise ValueError(f"Unsupported model name: {self.name}")

        # Drop the original fully connected layer
        org_resnet.fc = None  # Or you can set it to nn.Identity()

        # If you're loading weights manually, extract the state_dict
        org_resnet_state = org_resnet.state_dict()

        # Remove FC layer weights to avoid mismatch
        org_resnet_state.pop('fc.weight', None)
        org_resnet_state.pop('fc.bias', None)

        # Load into your model
        self.load_state_dict(org_resnet_state, strict=False)
        print("Initialized ResNet from torchvision with pretrained=True")