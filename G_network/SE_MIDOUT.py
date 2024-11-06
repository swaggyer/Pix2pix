
import torch.nn.functional as F

from attention.SENet import *
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

class SE_midout(nn.Module):
    "加入SE模块"
    def __init__(self,in_channels,out_channels):
        super(SE_midout,self).__init__()
        self.down_conv1 = _ConvBNReLU(in_channels=in_channels,out_channels=32,kernel_size=3,stride=1,padding=1,padding_mode = "reflect")
        self.down_conv2 = _DSConv(dw_channels=32,out_channels=64)
        self.down_conv3 = _DSConv(dw_channels=64,out_channels=128)
        self.down_conv4 = _DSConv(dw_channels=128,out_channels=256)
        # self.PyramidPooling = PyramidPooling(256,256)
        self.conv = nn.Sequential(nn.Conv2d(in_channels = 256,out_channels=256,kernel_size=1),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2))

        self.up_conv1 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=128,stride=1,kernel_size=3,padding=1,padding_mode="reflect"),
                                      nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(0.2)
                                      )

        self.up_conv2 = nn.Sequential(  nn.Conv2d(in_channels=256,out_channels=64,kernel_size=3,stride=1,padding =1,padding_mode="reflect"),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(0.2),
                                        nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(0.2))

        self.up_conv3 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=32,kernel_size=3,stride=1,padding =1,padding_mode="reflect"),
                                      nn.BatchNorm2d(32),
                                      nn.LeakyReLU(0.2),
                                        nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(0.2))
        self.up_conv4 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding =1,padding_mode="reflect"),
                                      nn.BatchNorm2d(32),
                                      nn.LeakyReLU(0.2),
                                        nn.Conv2d(in_channels=32,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=False),
                                        nn.BatchNorm2d(out_channels),

                                         nn.Tanh()
                                      )

        self.SE1 = SELayer(32)
        self.SE2 = SELayer(64)
        self.SE3 = SELayer(128)

        self.mid_out1 = nn.Sequential(nn.Conv2d(256,3,kernel_size=3,stride=1,padding=1),
                                      nn.BatchNorm2d(3),
                                      nn.Tanh())
        self.mid_out2 = nn.Sequential(nn.Conv2d(128,3,kernel_size=3,stride=1,padding=1),
                                      nn.BatchNorm2d(3),
                                      nn.Tanh())
        self.mid_out3 = nn.Sequential(nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1),
                                      nn.BatchNorm2d(3),
                                      nn.Tanh())

        self.last_conv = nn.Sequential(nn.Conv2d(12,3,kernel_size=3,stride=1,padding=1),
                                       nn.BatchNorm2d(3),
                                       nn.Tanh())



    def forward(self,x):
        h,w = x.size()[-2:]
        x1 = self.down_conv1(x)  #1*32*256*256
        x2 = self.SE1(x1)

        x3 = self.down_conv2(x1)  #1*64*128*128
        x4 = self.SE2(x3)

        x5 = self.down_conv3(x3)    #1*128*64*64
        x6 = self.SE3(x5)


        x7 = self.down_conv4(x5)    #1*256*32*32
        x8 = self.conv(x7)



        x9 = self.up_conv1(x8)
        x10 = torch.cat((x9,x6),dim = 1)

        x10_1 = self.mid_out1(x10)
        x10_1 = F.interpolate(x10_1, (h,w), mode='bicubic', align_corners=True)

        x11 = self.up_conv2(x10)   #  1*64*64*64
        x12 = torch.cat((x4,x11),dim=1)

        x12_1 = self.mid_out2(x12)
        x12_1 = F.interpolate(x12_1, (h,w), mode='bicubic', align_corners=True)

        x13 = self.up_conv3(x12)  #1*32*128*128
        x14 = torch.cat((x2,x13),dim=1)

        x14_1 = self.mid_out3(x14)
        x14_1 = F.interpolate(x14_1, (h,w), mode='bicubic', align_corners=True)
        x15 = self.up_conv4(x14)

        x16 = torch.cat((x10_1,x12_1,x14_1,x15),dim = 1)
        out = self.last_conv(x16)




        return out

if __name__ == '__main__':
    x = torch.randn(1,3,256,256)
    model = SE_midout(3,3)
    model.eval()
    y = model(x)
    print(y.shape)