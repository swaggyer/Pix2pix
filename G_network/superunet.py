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

class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]

        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class SuperUnet1(nn.Module):
    "先卷积降维再插值，减少计算量"
    def __init__(self,in_channels,out_channels):
        super(SuperUnet1,self).__init__()
        self.down_conv1 = _ConvBNReLU(in_channels=in_channels,out_channels=32,stride=2,padding=1,padding_mode = "reflect")
        self.down_conv2 = _DSConv(dw_channels=32,out_channels=64)
        self.down_conv3 = _DSConv(dw_channels=64,out_channels=128)
        self.PyramidPooling = PyramidPooling(128,128)
        self.up_conv1 = _DWConv(dw_channels=128,out_channels=32)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=3,padding=1,kernel_size=3,stride=1,padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

        )


    def forward(self,x):
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)

        x4 = self.PyramidPooling(x3)

        x5 = self.up_conv1(x4)
        x6 = F.interpolate(x5, scale_factor=4, mode='bilinear', align_corners=True)
        x7 = x1+x6
        x8 = self.up_conv2(x7)
        x9 = F.interpolate(x8,scale_factor=2,mode="bilinear",align_corners=True)
        x10 = nn.Tanh()(x9)
        return x10

class SuperUnet2(nn.Module):
    "U型结构，利用其它通道信息"
    def __init__(self,in_channels,out_channels):
        super(SuperUnet2,self).__init__()
        self.down_conv1 = _ConvBNReLU(in_channels=in_channels,out_channels=32,stride=2,padding=1,padding_mode = "reflect")
        self.down_conv2 = _DSConv(dw_channels=32,out_channels=64)
        self.down_conv3 = _DSConv(dw_channels=64,out_channels=128)
        self.PyramidPooling = PyramidPooling(128,128)
        self.up_conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.LeakyReLU(0.2)
                                      )
        self.conv1 = _DSConv(dw_channels=128,out_channels=64,stride=1)

        self.up_conv2 = nn.Sequential(  nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(0.2))
        self.conv2 = _DSConv(dw_channels=64,out_channels=32,stride=1)

        self.conv3 = nn.Sequential(nn.ConvTranspose2d(in_channels=32,out_channels = 3,kernel_size=4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.LeakyReLU(0.2)

        )


    def forward(self,x):
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)

        x4 = self.PyramidPooling(x3)
        x5 = self.up_conv1(x4)
        x6 = self.conv1(torch.cat((x5,x2),dim = 1))
        x7 = self.up_conv2(x6)
        x8 = self.conv2(torch.cat((x7,x1),dim = 1))
        x9 = self.conv3(x8)
        x10 = nn.Tanh()(x9)
        return x10

class SuperUnet3(nn.Module):
    "U型结构，取消跳跃连接"
    def __init__(self,in_channels,out_channels):
        super(SuperUnet3,self).__init__()
        self.down_conv1 = _ConvBNReLU(in_channels=in_channels,out_channels=32,stride=2,padding=1,padding_mode = "reflect")
        self.down_conv2 = _DSConv(dw_channels=32,out_channels=64)
        self.down_conv3 = _DSConv(dw_channels=64,out_channels=128)
        self.PyramidPooling = PyramidPooling(128,128)
        self.up_conv1 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=64,stride=1,kernel_size=3,padding=1,padding_mode="reflect"),
                                        nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),

                                      nn.BatchNorm2d(64),
                                      nn.LeakyReLU(0.2)
                                      )

        self.up_conv2 = nn.Sequential(  nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding =1,padding_mode="reflect"),
                                        nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU())

        self.up_conv3 = nn.Sequential(nn.ConvTranspose2d(in_channels=32,out_channels = 32,kernel_size=4,stride=2,padding=1,bias=False),
                                   nn.Conv2d(in_channels = 32,out_channels=out_channels,stride=1,kernel_size=3,padding = 1,padding_mode="reflect"),
                                        nn.BatchNorm2d(out_channels),
                                        nn.LeakyReLU(0.2)

        )





    def forward(self,x):
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)

        x4 = self.PyramidPooling(x3)
        x5 = self.up_conv1(x4)
        x6 = self.up_conv2(x5)
        x7 = self.up_conv3(x6)
        x8 = nn.Tanh()(x7)
        return x8


class Skip_attention(nn.Module):
    def __init__(self,inchannels):
        super(Skip_attention,self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=inchannels,out_channels=1,kernel_size=3,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
        )
        self.conv = _DWConv(inchannels,inchannels)


    def forward(self, x1,x2):
        feat = self.fc(x1)
        x = feat * x2
        out = self.conv(x)
        return out



class SuperUnet4(nn.Module):
    "跳跃链接使用金字塔池化"
    def __init__(self,in_channels,out_channels):
        super(SuperUnet4,self).__init__()
        self.down_conv1 = _ConvBNReLU(in_channels=in_channels,out_channels=32,stride=2,padding=1,padding_mode = "reflect")
        self.down_conv2 = _DSConv(dw_channels=32,out_channels=64)
        self.down_conv3 = _DSConv(dw_channels=64,out_channels=128)
        self.down_conv4 = _DSConv(dw_channels=128,out_channels=256)
        self.PyramidPooling = PyramidPooling(256,256)
        self.conv = _DWConv(dw_channels=256,out_channels=128)
        self.up_conv1 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=128,stride=1,kernel_size=3,padding=1,padding_mode="reflect"),
                                      nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(0.2)
                                      )

        self.up_conv2 = nn.Sequential(  nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding =1,padding_mode="reflect"),
                                        nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU())

        self.up_conv3 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding =1,padding_mode="reflect"),
                                        nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(0.2))
        self.up_conv4 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=out_channels,kernel_size=3,stride=1,padding =1,padding_mode="reflect"),
                                        nn.ConvTranspose2d(in_channels=out_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.LeakyReLU(0.2),
                                        nn.Tanh()
                                      )

        self.skip1 = Skip_attention(128)
        self.skip2 = Skip_attention(64)
        self.skip3 = Skip_attention(32)


    def forward(self,x):
        x1 = self.down_conv1(x)  #1*32*128*128
        x2 = self.down_conv2(x1)  #1*64*64*64
        x3 = self.down_conv3(x2)    #1*128*32*32
        x4 = self.down_conv4(x3)    #1*256*16*16


        x5 = self.PyramidPooling(x4) #1*256*16*16

        x6 = self.up_conv1(x5)   #1*128*32*32

        x7 = self.skip1(x3,x6)


        x8 = self.up_conv2(x7)   #  1*64*64*64
        x9 = self.skip2(x2,x8)

        x10 = self.up_conv3(x9)  #1*32*128*128
        x11 = self.skip3(x1,x10)

        out = self.up_conv4(x11)

        return out

class SuperUnet5(nn.Module):
    "比例跳跃链接"
    def __init__(self,in_channels,out_channels):
        super(SuperUnet5,self).__init__()
        self.down_conv1 = _ConvBNReLU(in_channels=in_channels,out_channels=32,stride=2,padding=1,padding_mode = "reflect")
        self.down_conv2 = _DSConv(dw_channels=32,out_channels=64)
        self.down_conv3 = _DSConv(dw_channels=64,out_channels=128)
        self.down_conv4 = _DSConv(dw_channels=128,out_channels=256)
        # self.PyramidPooling = PyramidPooling(256,256)
        self.conv = nn.Sequential(nn.Conv2d(in_channels = 256,out_channels=256,kernel_size=1))
        self.up_conv1 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=128,stride=1,kernel_size=3,padding=1,padding_mode="reflect"),
                                      nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(0.2)
                                      )

        self.up_conv2 = nn.Sequential(  nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding =1,padding_mode="reflect"),
                                        nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU())

        self.up_conv3 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding =1,padding_mode="reflect"),
                                        nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(32),
                                        nn.LeakyReLU(0.2))
        self.up_conv4 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=out_channels,kernel_size=3,stride=1,padding =1,padding_mode="reflect"),
                                        nn.ConvTranspose2d(in_channels=out_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1,bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.LeakyReLU(0.2),
                                        nn.Tanh()
                                      )

        self.skip1 = Skip_attention(128)
        self.skip2 = Skip_attention(64)
        self.skip3 = Skip_attention(32)


    def forward(self,x):
        x1 = self.down_conv1(x)  #1*32*128*128
        x2 = self.down_conv2(x1)  #1*64*64*64
        x3 = self.down_conv3(x2)    #1*128*32*32
        x5 = self.down_conv4(x3)    #1*256*16*16


        # x5 = self.PyramidPooling(x4) #1*256*16*16

        x6 = self.up_conv1(x5)   #1*128*32*32

        x7 = self.skip1(x3,x6)


        x8 = self.up_conv2(x7)   #  1*64*64*64
        x9 = self.skip2(x2,x8)

        x10 = self.up_conv3(x9)  #1*32*128*128
        x11 = self.skip3(x1,x10)

        out = self.up_conv4(x11)

        return out


class SuperUnet6(nn.Module):
    "加入SE模块"
    def __init__(self,in_channels,out_channels):
        super(SuperUnet6,self).__init__()
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
        self.up_conv4 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=out_channels,kernel_size=3,stride=1,padding =1,padding_mode="reflect"),
                                      nn.BatchNorm2d(out_channels),
                                      nn.LeakyReLU(0.2),
                                        # nn.ConvTranspose2d(in_channels=out_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1,bias=False),
                                        # nn.BatchNorm2d(out_channels),
                                        # nn.LeakyReLU(0.2),
                                        nn.Tanh()
                                      )

        self.SE1 = SELayer(32)
        self.SE2 = SELayer(64)
        self.SE3 = SELayer(128)



    def forward(self,x):
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


        x11 = self.up_conv2(x10)   #  1*64*64*64
        x12 = torch.cat((x4,x11),dim=1)

        x13 = self.up_conv3(x12)  #1*32*128*128
        x14 = torch.cat((x2,x13),dim=1)

        out = self.up_conv4(x14)


        return out


if __name__ =="__main__":
   x = torch.randn(1,3,256,256)
   layer = SuperUnet6(3,3)
   layer.eval()
   y = layer(x)
   print(y.shape)


