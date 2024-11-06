import torch
import torch.nn as nn
from utils.Transformer import *

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

# class CBHW(nn.Module):
#     def __init__(self):
#         super(CBHW,self).__init__()
#
#     def forward(self,x):
#         b = x.shape()[0]
#         c = x.shape()[1]
#         h = x.shape()[2]
#         w = x.shape()[3]
#         return[b,c,h,w]

class Trans_skip(nn.Module):

    def __init__(self,in_channels,out_channels):
        super(Trans_skip,self).__init__()

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
        self.input_dim1 = 256 ** 2 # 输入特征维度
        self.input_dim2 = 128 ** 2  # 输入特征维度
        self.input_dim3 = 64 ** 2  # 输入特征维度
        self.input_dim4 = 32**2

        # self.seq_length1 = 256 ** 2  # 序列长度
        # self.seq_length2 = 128 ** 2  # 序列长度
        # self.seq_length3 = 64 ** 2  # 序列长度
        # self.seq_length4 = 32**2

        depth = 3  # Transformer层数

        heads = 3  # 自注意力头数

        mlp_dim = 256  # 前馈网络的隐藏层维度

        dropout_rate = 0.1

        attn_dropout_rate = 0.1

        self.Trans_skip1 = TransformerModel(
            dim=self.input_dim1,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,

        )
        self.Trans_skip2 = TransformerModel(
            dim=self.input_dim2,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,

        )
        self.Trans_skip3 = TransformerModel(
            dim=self.input_dim3,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,

        )
        self.Trans_skip4 = TransformerModel(
            dim=self.input_dim4,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,

        )

    def forward(self,x):
        x1 = self.down_conv1(x)  #1*32*256*256
        x1_shape = x1.shape
        b1,c1,h1,w1 = x1_shape
        x2 = self.Trans_skip1(x1.view(b1,c1,h1*w1))
        x2 = x2.view(b1,c1,h1,w1)


        x3 = self.down_conv2(x1)  #1*64*128*128
        x3_shape = x3.shape
        b3, c3, h3, w3 = x3_shape
        x4 = self.Trans_skip2(x3.view(b3, c3, h3 * w3))
        x4 = x4.view(b3, c3, h3, w3)

        x5 = self.down_conv3(x3)    #1*128*64*64
        x5_shape = x5.shape
        b5, c5, h5, w5 = x5_shape
        x6 = self.Trans_skip3(x5.view(b5, c5, h5 * w5))
        x6 = x6.view(b5, c5, h5, w5)

        x7 = self.down_conv4(x5)    #1*256*32*32
        x7_shape = x7.shape
        b7, c7, h7, w7 = x7_shape
        x8 = self.Trans_skip4(x7.view(b7, c7, h7 * w7))
        x8 = x8.view(b7, c7, h7, w7)



        x9 = self.up_conv1(x8)
        x10 = torch.cat((x9,x6),dim = 1)


        x11 = self.up_conv2(x10)   #  1*64*64*64
        x12 = torch.cat((x4,x11),dim=1)

        x13 = self.up_conv3(x12)  #1*32*128*128
        x14 = torch.cat((x2,x13),dim=1)

        out = self.up_conv4(x14)


        return out

if __name__ == '__main__':
    x = torch.randn(1,3,256,256)
    layer = Trans_skip(3,3)
    layer.eval()
    y = layer(x)
    print(y.shape)