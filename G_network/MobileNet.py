import torch.nn as nn
import torch


# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)
class Depth_Separable_conv(nn.Module):
    def __init__(self,input_channels,output_channels,strides):
        super(Depth_Separable_conv,self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels = input_channels,
                      out_channels = input_channels,
                      kernel_size=(3,3),
                      stride=(strides,strides),
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
                      kernel_size=(1,1),
                      stride=(1,1),
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
            nn.ReLU()
        )
    def forward(self,x):
        return  self.conv_block(x)
class MobileNet(nn.Module):
    def __init__(self,input_channels,output_channels):
        super( MobileNet,self).__init__()
        self.conv_block = Conv_block(input_channels,32)
        self.down1 = Depth_Separable_conv(32,64,strides=2)
        self.down2 = Depth_Separable_conv(64,128,strides=2)
        self.down3 = Depth_Separable_conv(128,256,strides=2)
        self.down4 = Depth_Separable_conv(256,512,strides=2)
        self.up1 = Up_block(512,256)
        self.up2 = Up_block(256,128)
        self.up3 = Up_block(128,64)
        self.up4 = Up_block(64,32)
        self.final = nn.Conv2d(32,output_channels,stride=1,kernel_size=3,padding=1,padding_mode="reflect")
        self.tanh = nn.Tanh()

    def forward(self,x):
        x1 = self.conv_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5)
        x7 = self.up2(x6)
        x8 = self.up3(x7)
        x9 = self.up4(x8)
        x10 = self.final(x9)
        x11 = self.tanh(x10)
        return x11



if __name__ == "__main__":
    x = torch.randn(1,3,256,256)
    print(x.type)
    layer = MobileNet(3,3)

    y = layer(x)
    print(y.shape)
