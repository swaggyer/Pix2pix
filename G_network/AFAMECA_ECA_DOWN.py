
from Utils.AFAM import *
from Utils.AFAM_Better import AFAM_Module_ECA
from Utils.CACB import *
from Utils.SPPCSPC import *
from attention.ECA import *
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

class AFAMECA_ECA_DOWN(nn.Module):
    "在UNet加入AFAM，自适应特征注意力融合，密集跳跃连接的一种表现形式"
    def __init__(self,in_channels,out_channels):
        super(AFAMECA_ECA_DOWN,self).__init__()
        self.down_conv1 = CACB_Module(in_ch=in_channels,out_ch=32,last=False)
        self.down_conv2 = _DSConv(dw_channels=32,out_channels=64)
        self.down_conv3 = _DSConv(dw_channels=64,out_channels=128)
        self.down_conv4 = _DSConv(dw_channels=128,out_channels=256)
        # self.PyramidPooling = PyramidPooling(256,256)
        self.conv = nn.Sequential(nn.Conv2d(in_channels = 256,out_channels=256,kernel_size=1),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2))

        self.selay1 = ECA_layer(32)
        self.selay2 = ECA_layer(64)
        self.selay3 = ECA_layer(128)


        self.AFAM1 = AFAM_Module_ECA(out_ch=128,in_chs = [32,64,128,256])
        self.AFAM2 = AFAM_Module_ECA(out_ch=64,in_chs = [32,64,128,128])
        self.AFAM3 = AFAM_Module_ECA(out_ch=32, in_chs=[32, 64, 128, 64])
        self.lastconv = CACB_Module(in_ch=32,out_ch=out_channels,last=True)


    def forward(self,x):
        x1 = self.down_conv1(x)  #1*32*256*256
        x11 = self.selay1(x1)

        x2 = self.down_conv2(x11)  #1*64*128*128
        x22 = self.selay2(x2)

        x3 = self.down_conv3(x22)    #1*128*64*64
        x33 = self.selay3(x3)

        x4 = self.down_conv4(x33)    #1*256*32*32


        x5 = self.conv(x4)

        x6 = self.AFAM1(x11,x22,x33,x5)

        x7 = self.AFAM2(x11,x22,x33,x6)

        x8 = self.AFAM3(x11,x22,x33,x7)

        out = self.lastconv(x8)

        return out







if __name__ =="__main__":
   x = torch.randn(1,3,256,256)
   layer = AFAMECA_ECA_DOWN(3,3)
   layer.eval()
   y = layer(x)
   print(y.shape)