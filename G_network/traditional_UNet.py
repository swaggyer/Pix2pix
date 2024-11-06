import torch.nn as nn
import torch



class ConvBlock(nn.Module):
    def __init__(self,ch_input,ch_output):
        super(ConvBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_input,ch_output,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(num_features=ch_output),
            nn.LeakyReLU(),
            nn.Conv2d(ch_output,ch_output,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(num_features=ch_output),
            nn.LeakyReLU(),
        )
    def forward(self,x):
        return self.block(x)





class UNetL4_Pool(nn.Module):
    def __init__(self,ch_input,ch_output):
        super(UNetL4_Pool,self).__init__()
        self.ch_input = ch_input
        self.ch_output = ch_output
        self.pool = nn.MaxPool2d(stride=(2,2),kernel_size=2)

        self.down1 = ConvBlock(ch_input,64)
        self.down2 = ConvBlock(64,128)
        self.down3 =ConvBlock(128,256)
        self.down4 = ConvBlock(256,512)
        self.down5 = ConvBlock(512,1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, padding_mode="zeros")
        self.up1_1 = ConvBlock(1024,512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, padding_mode="zeros")
        self.up2_1 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, padding_mode="zeros")
        self.up3_1 = ConvBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, padding_mode="zeros")
        self.up4_1 = ConvBlock(128, 64)

        self.finalconv = nn.Sequential(nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1,padding_mode="zeros"),
                                       nn.Tanh())
    def forward(self, x):
        x1 = self.down1(x)

        x2 = self.down2(self.pool(x1))

        x3 = self.down3(self.pool(x2))

        x4 = self.down4(self.pool(x3))

        x5 = self.down5(self.pool(x4))

        x6 = self.up1(x5)
        x6 =torch.cat((x6,x4),dim = 1)

        x7 = self.up2(self.up1_1(x6))
        x7 = torch.cat((x7,x3),dim = 1)

        x8 = self.up3(self.up2_1(x7))
        x8 = torch.cat((x8, x2), dim=1)

        x9 = self.up4(self.up3_1(x8))
        x9 = torch.cat((x9, x1), dim=1)

        x10 = self.up4_1(x9)
        x10 =self.finalconv(x10)

        return x10

class UNetL4_Conv(nn.Module):
    def __init__(self,ch_input,ch_output):
        super(UNetL4_Conv,self).__init__()
        self.ch_input = ch_input
        self.ch_output = ch_output
        self.pool = nn.MaxPool2d(stride=(2,2),kernel_size=2)
        self.downconv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.downconv2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4,stride=2,padding=1)
        self.downconv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.downconv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.down1 = ConvBlock(ch_input,64)
        self.down2 = ConvBlock(64,128)
        self.down3 =ConvBlock(128,256)
        self.down4 = ConvBlock(256,512)
        self.down5 = ConvBlock(512,1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, padding_mode="zeros")
        self.up1_1 = ConvBlock(1024,512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, padding_mode="zeros")
        self.up2_1 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, padding_mode="zeros")
        self.up3_1 = ConvBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, padding_mode="zeros")
        self.up4_1 = ConvBlock(128, 64)

        self.finalconv = nn.Sequential(nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1,padding_mode="zeros"),
                                       nn.Tanh())
    def forward(self, x):
        x1 = self.down1(x)

        x2 = self.down2(self.downconv1(x1))

        x3 = self.down3(self.downconv2(x2))

        x4 = self.down4(self.downconv3(x3))

        x5 = self.down5(self.downconv4(x4))

        x6 = self.up1(x5)
        x6 =torch.cat((x6,x4),dim = 1)

        x7 = self.up2(self.up1_1(x6))
        x7 = torch.cat((x7,x3),dim = 1)

        x8 = self.up3(self.up2_1(x7))
        x8 = torch.cat((x8, x2), dim=1)

        x9 = self.up4(self.up3_1(x8))
        x9 = torch.cat((x9, x1), dim=1)

        x10 = self.up4_1(x9)
        x10 =self.finalconv(x10)

        return x10





if __name__ =="__main__":
    x = torch.randn(1,3,256,256)
    model = UNetL4_Conv(ch_input=3,ch_output=3)
    y = model(x)
    print(y.shape)