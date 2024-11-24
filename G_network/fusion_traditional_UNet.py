from pydoc import replace
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.AFAM_Better import *

class DoubleConv_down(nn.Sequential):
    def __init__(self, in_channels, out_channels):

        super(DoubleConv_down, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        output = self.block(input)
        return output

class UNetL3_Tconv_fusion(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UNetL3_Tconv_fusion,self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_ch,out_channels=32,kernel_size=3,stride=1,padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(inplace=True) )
        self.down1 = DoubleConv_down(in_channels=32,out_channels=64)
        self.down2 = DoubleConv_down(in_channels=64,out_channels=128)
        self.down3 = DoubleConv_down(in_channels=128,out_channels=256)

        self.middle_conv = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1))


        self.fusion1 = AFAM_Module_ECA(out_ch = 128,in_chs=[32,64,128,256])
        self.fusion2 = AFAM_Module_ECA(out_ch=64, in_chs=[32, 64, 128, 128])
        self.fusion3 = AFAM_Module_ECA(out_ch=32, in_chs=[32, 64, 128, 64])

        self.lastconv = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=self.out_ch,kernel_size=3,stride=1,padding=1),
                                      nn.BatchNorm2d(3),
                                      nn.ReLU(inplace=True),)

        self.act = nn.Tanh()

    def forward(self,x):
        x1= self.first_conv(x)  #--256*256*32
        x2 = self.down1(x1)  #--128*128*64
        x3 = self.down2(x2)  #--64*64*128
        x4 = self.down3(x3)  #--32*32*256

        x5 = self.middle_conv(x4)  #--32*32*256

        x6 = self.fusion1(x1,x2,x3,x5)
        x7 = self.fusion2(x1,x2,x3,x6)
        x8 = self.fusion3(x1,x2,x3,x7)

        x9 = self.lastconv(x8)

        output = self.act(x9)

        return output



class UNetL3_Tconv_fusion_noact(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UNetL3_Tconv_fusion_noact,self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_ch,out_channels=32,kernel_size=3,stride=1,padding=1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(inplace=True) )
        self.down1 = DoubleConv_down(in_channels=32,out_channels=64)
        self.down2 = DoubleConv_down(in_channels=64,out_channels=128)
        self.down3 = DoubleConv_down(in_channels=128,out_channels=256)

        self.middle_conv = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1))


        self.fusion1 = AFAM_Module_ECA(out_ch = 128,in_chs=[32,64,128,256])
        self.fusion2 = AFAM_Module_ECA(out_ch=64, in_chs=[32, 64, 128, 128])
        self.fusion3 = AFAM_Module_ECA(out_ch=32, in_chs=[32, 64, 128, 64])

        self.lastconv = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=self.out_ch,kernel_size=3,stride=1,padding=1),
                                      nn.BatchNorm2d(3),
                                      nn.ReLU(inplace=True),)

        # self.act = nn.Tanh()

    def forward(self,x):
        x1= self.first_conv(x)  #--256*256*32
        x2 = self.down1(x1)  #--128*128*64
        x3 = self.down2(x2)  #--64*64*128
        x4 = self.down3(x3)  #--32*32*256

        x5 = self.middle_conv(x4)  #--32*32*256

        x6 = self.fusion1(x1,x2,x3,x5)
        x7 = self.fusion2(x1,x2,x3,x6)
        x8 = self.fusion3(x1,x2,x3,x7)

        x9 = self.lastconv(x8)

        # output = self.act(x9)

        return x9


class UNetL3_Tconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetL3_Tconv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.down1 = DoubleConv_down(in_channels=32, out_channels=64)
        self.down2 = DoubleConv_down(in_channels=64, out_channels=128)
        self.down3 = DoubleConv_down(in_channels=128, out_channels=256)

        self.middle_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))

        self.up1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 )

        self.upconv1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))

        self.up2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), )

        self.upconv2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.up3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), )

        self.upconv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True), )

        self.act = nn.Tanh()

    def forward(self, x):
        x1 = self.first_conv(x)  # --256*256*32
        x2 = self.down1(x1)  # --128*128*64
        x3 = self.down2(x2)  # --64*64*128
        x4 = self.down3(x3)  # --32*32*256

        x5 = self.middle_conv(x4)  # --32*32*256

        x6 = self.up1(x5)  # --64*64*128
        x7 = torch.cat((x3, x6), dim=1)  # --64*64*256
        x8 = self.upconv1(x7)  ##--64*64*128

        x8 = self.up2(x8)  # --128*128*64
        x9 = torch.cat((x8, x2),dim =1)  # --128*128*128
        x10 = self.upconv2(x9)  # --128*128*64

        x11 = self.up3(x10)  # --256*256*32
        x12 = torch.cat((x11, x1),dim =1)  # --256*256*64
        x13 = self.upconv3(x12)

        x14 = self.lastconv(x13)

        output = self.act(x14)

        return output

class UNetL3_Tconv_noact(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetL3_Tconv_noact, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.down1 = DoubleConv_down(in_channels=32, out_channels=64)
        self.down2 = DoubleConv_down(in_channels=64, out_channels=128)
        self.down3 = DoubleConv_down(in_channels=128, out_channels=256)

        self.middle_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))

        self.up1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 )

        self.upconv1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))

        self.up2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), )

        self.upconv2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.up3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), )

        self.upconv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True), )



    def forward(self, x):
        x1 = self.first_conv(x)  # --256*256*32
        x2 = self.down1(x1)  # --128*128*64
        x3 = self.down2(x2)  # --64*64*128
        x4 = self.down3(x3)  # --32*32*256

        x5 = self.middle_conv(x4)  # --32*32*256

        x6 = self.up1(x5)  # --64*64*128
        x7 = torch.cat((x3, x6), dim=1)  # --64*64*256
        x8 = self.upconv1(x7)  ##--64*64*128

        x8 = self.up2(x8)  # --128*128*64
        x9 = torch.cat((x8, x2),dim =1)  # --128*128*128
        x10 = self.upconv2(x9)  # --128*128*64

        x11 = self.up3(x10)  # --256*256*32
        x12 = torch.cat((x11, x1),dim =1)  # --256*256*64
        x13 = self.upconv3(x12)

        x14 = self.lastconv(x13)



        return x14


if __name__ == '__main__':
    x = torch.randn(1,3,256,256)
    layer = UNetL3_Tconv_noact(3,3)
    y = layer(x)
    print(y.shape)

