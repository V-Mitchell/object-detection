from torch import nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, first_block=False):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn0 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if first_block:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels*self.expansion, 1, stride, 0)
                                            , nn.BatchNorm2d(out_channels*self.expansion))
    
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample:
            identity = self.downsample(identity)
        
        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock = Bottleneck
                 , blocks_list=[3, 4, 6, 3]
                 , out_channels_list=[64, 128, 256, 512]
                 , num_channels=3):
        super().__init__()

        self.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self.create_layer(ResBlock, blocks_list[0], 64, out_channels_list[0], 1)
        self.layer2 = self.create_layer(ResBlock, blocks_list[1], out_channels_list[0]*ResBlock.expansion, out_channels_list[1], 2)
        self.layer3 = self.create_layer(ResBlock, blocks_list[2], out_channels_list[1]*ResBlock.expansion, out_channels_list[2], 2)
        self.layer4 = self.create_layer(ResBlock, blocks_list[3], out_channels_list[2]*ResBlock.expansion, out_channels_list[3], 2)

    def create_layer(self, ResBlock, num_blocks, in_channels, out_channels, stride=1):
        layer = []
        for i in range(num_blocks):
            if i == 0:
                layer.append(ResBlock(in_channels, out_channels, stride, first_block=True))
            else:
                layer.append(ResBlock(out_channels*ResBlock.expansion, out_channels))
        
        return nn.Sequential(*layer)


    def forward(self, x):
        x = self.max_pool(self.relu(self.bn0(self.conv0(x))))

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return (f1, f2, f3, f4)

BLOCKS = {"Bottleneck": Bottleneck}
