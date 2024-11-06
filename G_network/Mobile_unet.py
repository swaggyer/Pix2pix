import torch.nn as nn
import torch

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Depth_Separable_conv(nn.Module):
    def __init__(self,input_channels,output_channels,strides):
        super(Depth_Separable_conv,self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels = input_channels,
                      out_channels = input_channels,
                      kernel_size=3,
                      stride=strides,
                      groups= input_channels,
                      padding=(1,1),
                      padding_mode="reflect"
                      ),
            nn.BatchNorm2d(num_features =input_channels ),
            nn.ReLU6()
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels= input_channels,
                      out_channels= output_channels,
                      kernel_size=1,
                      stride=1,
                      groups=1,
                      padding=(0,0)
                      ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6()
        )
    def forward(self,x):
        x1 = self.depthwise_conv(x)
        output = self.pointwise_conv(x1)
        return output

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


class Mobile_UNetL4(nn.Module):
    def __init__(self,ch_input,ch_output):
        super(Mobile_UNetL4,self).__init__()

        self.conv1 = Conv_block(ch_input,32)
        self.down1 = Depth_Separable_conv(32,64,strides=2)
        self.down2 = Depth_Separable_conv(64,128,strides=2)
        self.down3 = Depth_Separable_conv(128,256,strides=2)
        self.down4 = Depth_Separable_conv(256,512,strides=2)


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


##以下为膨胀卷积版本
class Conv_block_dilation(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(Conv_block_dilation, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=2,
                      padding_mode="reflect",dilation=2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6()
        )
    def forward(self,x):
        return self.conv_block(x)

class Up_block_dilation(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(Up_block_dilation,self).__init__()
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
                      padding=2,
                      padding_mode="reflect",
                      dilation=2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6()
        )
    def forward(self,x):
        output = self.layers(x)
        return output

class Depth_Separable_conv_dilation(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(Depth_Separable_conv_dilation,self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels = input_channels,
                      out_channels = input_channels,
                      kernel_size=3,
                      groups= input_channels,
                      padding=3,
                      padding_mode="reflect",
                      dilation=3


                      ),
            nn.BatchNorm2d(num_features =input_channels ),
            nn.ReLU6()
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels= input_channels,
                      out_channels= output_channels,
                      kernel_size=1,
                      stride=1,
                      groups=1,
                      padding=0,

                      ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(),
            nn.MaxPool2d(2),
        )
    def forward(self,x):
        x1 = self.depthwise_conv(x)
        output = self.pointwise_conv(x1)
        return output


class Mobile_UNetL4_dilation(nn.Module):
    def __init__(self,ch_input,ch_output):
        super(Mobile_UNetL4_dilation,self).__init__()

        self.conv1 = Conv_block_dilation(ch_input,32)
        self.down1 = Depth_Separable_conv_dilation(32,64)
        self.down2 = Depth_Separable_conv_dilation(64,128)
        self.down3 = Depth_Separable_conv_dilation(128,256)
        self.down4 = Depth_Separable_conv_dilation(256,512)


        self.up1 = Up_block_dilation(512, 256)
        self.up2 = Up_block_dilation(256, 128)
        self.up3 = Up_block_dilation(128, 64)
        self.up4 = Up_block_dilation(64, 32)

        self.conv2 = Conv_block_dilation(512,256)
        self.conv3 = Conv_block_dilation(256,128)
        self.conv4 = Conv_block_dilation(128,64)
        self.conv5 = Conv_block_dilation(64, 32)

        self.finalconv = nn.Sequential(nn.Conv2d(32,ch_output,kernel_size=3,stride=1,padding=2,dilation=2,padding_mode="reflect"),
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
    layer =Mobile_UNetL4_dilation(3,3)
    y = layer(x)
    print(y.shape)