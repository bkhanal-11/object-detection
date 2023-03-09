import torch
import torch.nn as nn
from torchsummary import summary

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm = True, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=not batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.batch_norm = batch_norm

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x))) if self.batch_norm else self.conv(x)

class DarknetResidualBlock(nn.Module):
    def __init__(self, in_channels, use_res = True, num_repeat = 1):
        super().__init__()

        mid_channels = in_channels // 2
        self.layers = nn.ModuleList([nn.Sequential(
                ConvolutionalBlock(in_channels, mid_channels, kernel_size=1),
                ConvolutionalBlock(mid_channels, in_channels, kernel_size=3, padding=1)
            ) for _ in range(num_repeat)])
        
        self.use_res = use_res
        self.num_repeat = num_repeat

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) if self.use_res else layer(x)

        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.prediction = nn.Sequential(
            ConvolutionalBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            ConvolutionalBlock(2 * in_channels, 3 * (num_classes + 5), batch_norm=False, kernel_size=1)
        )

        self.num_classes = num_classes

    def forward(self, x):
        return self.prediction(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
    
class Darknet53(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(Darknet53, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._darknet53_layers()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _darknet53_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    ConvolutionalBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(DarknetResidualBlock(in_channels, num_repeat=num_repeats))
            elif isinstance(module, str):
                if module == "S":
                    break
            
        return layers

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, DarknetResidualBlock) and layer.num_repeat == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(ConvolutionalBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size == 3 else 0))
                in_channels = out_channels

            elif isinstance(module, list) and module[0] == "B":
                num_repeat = module[1]
                layers.append(DarknetResidualBlock(in_channels, num_repeat=num_repeat))

            elif isinstance(module, str) and module == "S":
                layers.append(DarknetResidualBlock(in_channels, use_res=False, num_repeat=1))
                layers.append(ConvolutionalBlock(in_channels, in_channels//2, kernel_size=1))
                layers.append(ScalePrediction(in_channels//2, num_classes=self.num_classes))
                in_channels = in_channels // 2

            elif isinstance(module, str) and module == "U":
                layers.append(nn.Upsample(scale_factor=2))
                in_channels = in_channels * 3

            else:
                raise ValueError(f"Invalid module {module}")

        return layers

if __name__ == "__main__":
    # Create an instance of the model
    # model = Darknet53(num_classes=1000)

    # # Generate some random input tensor
    # input_tensor = torch.randn(1, 3, 224, 224)

    # # Pass the input through the model
    # output = model(input_tensor)

    # # Print the shape of the output
    # print(output.shape)

    # print(summary(model.to('cuda'), (3, 256, 256)))
    model = YOLOv3()
    summary(model.to('cuda'), (3, 416, 416))
