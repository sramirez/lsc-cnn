from torchvision.models import ResNet
from torch.utils.model_zoo import load_url as load_state_dict_from_url

class ResnetBackbone(ResNet):
    def __init__(self, *args):
        super(ResnetBackbone, self).__init__(self, *args)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        y = self.layer2(x)
        z = self.layer3(y)
        v = self.layer4(z)

        return x, y, z, v


def resnetBackbone():
    model = ResnetBackbone(ResNet.Bottleneck, [3, 4, 23, 3])
    model.load_state_dict(load_state_dict_from_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'))
    return model