import torch
import torch.nn as nn
from torchsummary import summary

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvolutionalBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)

        return x

class DarknetBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarknetBlock, self).__init__()

        mid_channels = in_channels // 2

        self.conv1 = ConvolutionalBlock(in_channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = ConvolutionalBlock(mid_channels, in_channels, kernel_size=3)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + identity

        return x

class Darknet53(nn.Module):
    def __init__(self, num_classes=1000):
        super(Darknet53, self).__init__()

        self.conv1 = ConvolutionalBlock(3, 32, kernel_size=3, padding=1)
        self.layer1 = self._make_layer(64, 1)
        self.layer2 = self._make_layer(128, 2)
        self.layer3 = self._make_layer(256, 8)
        self.layer4 = self._make_layer(512, 8)
        self.layer5 = self._make_layer(1024, 4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, out_channels, num_blocks):
        layers = []

        layers.append(ConvolutionalBlock(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, stride=2, padding=1))

        for _ in range(num_blocks):
            layers.append(DarknetBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    # Create an instance of the model
    model = Darknet53(num_classes=1000)

    # # Generate some random input tensor
    # input_tensor = torch.randn(1, 3, 224, 224)

    # # Pass the input through the model
    # output = model(input_tensor)

    # # Print the shape of the output
    # print(output.shape)

    print(summary(model.to('cuda'), (3, 256, 256)))