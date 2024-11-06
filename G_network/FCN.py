import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN8s(nn.Module):
    def __init__(self, num_classes=3):
        super(FCN8s, self).__init__()
        # num_classes includes background, for PASCAL VOC it's 20+1  
        self.layer1 = self._make_block(2, 3, 64)
        self.layer2 = self._make_block(2, 64, 128)
        self.layer3 = self._make_block(3, 128, 256)
        self.layer4 = self._make_block(3, 256, 512)
        self.layer5 = self._make_block(3, 512, 512)

        # The two convolutional layers replace the original fully connected layers in VGG  
        mid_channels = 1024
        self.conv6 = nn.Conv2d(512, mid_channels, kernel_size=7, padding=3)
        self.conv7 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)

        # 1x1 convolutions to change the number of channels for pooling layers  
        self.score32 = nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        self.score16 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score8 = nn.Conv2d(256, num_classes, kernel_size=1)

        # Transposed convolutions for upsampling feature maps  
        self.up_sample8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4,
                                              output_padding=8 - 16 // 2)  # Adjust output_padding to match PaddlePaddle's behavior
        self.up_sample16x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.up_sample32x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)


        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_block(self, num, in_channels, out_channels, padding=1):
        """Constructs a block of convolutions and ReLUs, followed by a max pool."""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch_size, 3, height, width]  
        out = self.layer1(x)  # [batch_size, 64, height/2, width/2]  
        out = self.layer2(out)  # [batch_size, 128, height/4, width/4]  
        pool3 = self.layer3(out)  # [batch_size, 256, height/8, width/8]  
        pool4 = self.layer4(pool3)  # [batch_size, 512, height/16, width/16]  
        pool5 = self.layer5(pool4)  # [batch_size, 512, height/32, width/32]  
        x = self.conv6(pool5)  # [batch_size, mid_channels, height/32, width/32]  
        x = self.conv7(x)  # [batch_size, mid_channels, height/32, width/32]  
        score32 = self.score32(x)  # [batch_size, num_classes, height/32, width/32]  

        up_pool16 = self.up_sample32x(
            score32)  # [batch_size, num_classes, height/16, width/16] (adjusted for stride and padding)
        score16 = self.score16(pool4)  # [batch_size, num_classes, height/16, width/16]  
        fuse_16 = up_pool16 + score16  # Element-wise addition  

        up_pool8 = self.up_sample16x(fuse_16)  # [batch_size, num_classes, height/8, width/8]  
        score8 = self.score8(pool3)  # [batch_size, num_classes, height/8, width/8]  
        fuse_8 = up_pool8 + score8  # Element-wise addition  
        heatmap = self.up_sample8x(
            fuse_8)  # [batch_size, num_classes, height, width] (adjusted for stride, padding, and output_padding)

        return heatmap

if __name__ == '__main__':
    x = torch.randn(1,3,256,256)
    layer = FCN8s()
    y = layer(x)
    print(y.shape)