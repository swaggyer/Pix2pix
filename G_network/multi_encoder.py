import torch
import torch.nn as nn
import torch.nn.functional as F
class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Up_block(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(Up_block,self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_channels,
                               out_channels=input_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               padding_mode="zeros"
                               ),
            nn.Conv2d(in_channels=input_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode="reflect"),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6()
        )
    def forward(self,x):
        output = self.layers(x)
        return output

class _DSConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=2):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    "分组卷积 减少参数量 提速 "
    def __init__(self, dw_channels, out_channels, stride=1):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Multi_attention(nn.Module):
    def __init__(self,inchannels = 3,outchannels = 3):
        super(Multi_attention,self).__init__()
        self.first_conv = _ConvBNReLU(in_channels=inchannels,out_channels=32,padding=1)
        self.down1 = _DSConv(dw_channels=32,out_channels=64)
        self.down2 = _DSConv(dw_channels=64,out_channels=128)
        self.down3 = _DSConv(dw_channels=128,out_channels=256)

        self.up1 = Up_block(256,128)
        self.up2 = Up_block(128,64)
        self.up3 = Up_block(64,32)

        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=outchannels,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(outchannels),
            nn.Tanh()

        )

        self.conv1 =_DSConv(dw_channels=256,out_channels=128,stride=1)
        self.conv2 = _DSConv(dw_channels=320, out_channels=64,stride=1)
        self.conv3 = _DSConv(dw_channels=288, out_channels=32,stride=1)





    def forward(self,x):
        x1 = self.first_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)


        # x5 = F.interpolate(x4, scale_factor=2, mode='bicubic', align_corners=True)
        x6 = F.interpolate(x4, scale_factor=4, mode='bicubic', align_corners=True)
        x7 = F.interpolate(x4, scale_factor=8, mode='bicubic', align_corners=True)


        x8 = self.up1(x4)
        x9 = self.conv1(torch.cat((x8,x3),dim = 1))



        x10 = self.up2(x9)
        x11 = self.conv2(torch.cat((x10,x6),dim = 1))


        x12 = self.up3(x11)
        x13 = self.conv3(torch.cat((x12,x7),dim = 1))

        x14 = self.last_conv(x13)

        return x14

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = Multi_attention(3,3)
    model.eval()
    y = model(x)
    print(y.shape)
