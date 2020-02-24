from torchvision.models import ResNet
import torch

class ResnetBackbone(ResNet):
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        y = self.layer2(x)
        z = self.layer3(x)
        v = self.layer4(x)

        return x, y, z, v