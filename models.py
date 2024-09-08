import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.AvgPool2d(kernel_size=28)
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

def get_resnet18(num_classes, pretrained=False):
    model = resnet18(pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    return model
