
import torch
import torch.nn as nn
class Residual_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Residual_block, self).__init__()
        self.main_residual = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels // 2, stride=1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels // 2,out_channels= input_channels // 2, stride=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels // 2,out_channels= output_channels, stride=1, kernel_size=1)

        )
        self.minor_residual = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.main_residual(x)
        x2 = self.minor_residual(x)
        x =torch.relu(x1 + x2)

        return x

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

class Conv_block(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(Conv_block, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1,
                      padding_mode="reflect"),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6()
        )
    def forward(self,x):
        return self.conv_block(x)


class Residual_UNetL4(nn.Module):
    def __init__(self,ch_input,ch_output):
        super(Residual_UNetL4,self).__init__()

        self.conv1 = Conv_block(ch_input,32)
        self.down1 = Residual_block(32,64)
        self.down2 = Residual_block(64,128)
        self.down3 = Residual_block(128,256)
        self.down4 =Residual_block(256,512)


        self.up1 = Up_block(512, 256)
        self.up2 = Up_block(256, 128)
        self.up3 = Up_block(128, 64)
        self.up4 = Up_block(64, 32)

        self.conv2 = Conv_block(512,256)
        self.conv3 = Conv_block(256,128)
        self.conv4 = Conv_block(128,64)
        self.conv5 = Conv_block(64, 32)

        self.finalconv = nn.Sequential(nn.Conv2d(32,ch_output,kernel_size=3,stride=1,padding=1,padding_mode="zeros"),
                                       nn.BatchNorm2d(ch_output),
                                       nn.ReLU6(),
                                       nn.Tanh())
    def forward(self, x):

        x1 = self.conv1(x) #32*256*256

        x2 = self.down1(x1)#64*128*128

        x3 = self.down2(x2)#128*64*64

        x4 = self.down3(x3)#256*32*32

        x5 = self.down4(x4)#512*16*16





        x6 = self.up1(x5)#256*32*32

        x7 = self.conv2(torch.cat((x6,x4),dim=1))#512*32*32 ---256*32*32
        x8 = self.up2(x7)#128*64*64

        x9 = self.conv3(torch.cat((x8,x3),dim=1))

        x10 = self.up3(x9)#64*128*128
        x11 =self.conv4(torch.cat((x10,x2),dim = 1))

        x12 = self.up4(x11)#32*256*256
        x13 =self.conv5(torch.cat((x12,x1),dim = 1))

        x14 = self.finalconv(x13)


        return x14

if __name__ =="__main__":
    x = torch.randn(1,3,256,256)
    module = Residual_UNetL4(3,3)
    y = module(x)
    print(y.shape)
