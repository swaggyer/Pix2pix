import torch
from cv2.typing import map_int_and_double
from torch import nn



class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        ## block = [pad + conv + norm + relu + pad + conv + norm]
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),  ## ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, 3),  ## 卷积
            nn.BatchNorm2d(in_features),  ## InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
            nn.ReLU(inplace=True),  ## 非线性激活
            nn.ReflectionPad2d(1),  ## ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, 3),  ## 卷积
            nn.BatchNorm2d(in_features),  ## InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
        )

    def forward(self, x):
        return x + self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.residual = ResidualBlock(in_channels)
        self.conv_out = nn.Conv2d(in_channels, out_channels, 1)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)



    def forward(self, x):
        residual_out = self.residual(x)
        out = self.conv_out(residual_out)
        out = self.bn_out(out)
        out = self.leaky_relu(out)
        return out


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # VGGBlock实际上就是相当于做了两次卷积
        out = self.conv1(x)
        out = self.bn1(out)     # 归一化
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out



class ResUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4])

        self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)




    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


if __name__ == '__main__':
    x = torch.randn(1,3,256,256)
    module = ResBlock(3,3)
    y = module(x)
    print(y.shape)