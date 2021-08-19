import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
activation = nn.ReLU
gnorm = nn.GroupNorm
norm = nn.BatchNorm2d

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=2):
        super(U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """

    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class R2U_Net_3(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """

    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net_3, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(img_ch, filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[1], output_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        # e1 = self.RRCNN1(x)

        # e2 = self.Maxpool(e1)

        e2 = self.RRCNN2(x)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d4 = self.Up5(e5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.Up_RRCNN5(d4)

        d3 = self.Up4(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Up_RRCNN4(d3)

        d2 = self.Up3(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_RRCNN3(d2)

        # d2 = self.Up2(d3)
        # d2 = torch.cat((e1, d2), dim=1)
        # d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        # out = self.active(out)

        return out


class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """
    def __init__(self, img_ch=1, output_ch=2, t=3):
        super(R2U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      # out = self.active(out)

        return out

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net_3(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net_3, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(img_ch, filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        # self.Up2 = up_conv(filters[1], filters[0])
        # self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        # self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[1], output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        # e1 = self.Conv1(x)

        # e2 = self.Maxpool1(e1)
        e2 = self.Conv2(x)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d4 = self.Up5(e5)
        # print(d5.shape)

        # 改到这里了，后面还没完全搞懂，不慌先
        x4 = self.Att5(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv5(d4)

        d3 = self.Up4(d4)
        x3 = self.Att4(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv4(d3)

        d2 = self.Up3(d3)
        x2 = self.Att3(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv3(d2)

        # d2 = self.Up2(d3)
        # x1 = self.Att2(g=d2, x=e1)
        # d2 = torch.cat((x1, d2), dim=1)
        # d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out

class AttU_Net(nn.Module):

    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out


class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """
    def __init__(self, in_ch=3, out_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out

class R2AttU_Net_3(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """

    def __init__(self, in_ch=3, out_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(in_ch, filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        # e1 = self.RRCNN1(x)

        # e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(x)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d4 = self.Up5(e5)
        e4 = self.Att5(g=d4, x=e4)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.Up_RRCNN5(d4)

        d3 = self.Up4(d4)
        e3 = self.Att4(g=d3, x=e3)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Up_RRCNN4(d3)

        d2 = self.Up3(d3)
        e2 = self.Att3(g=d2, x=e2)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_RRCNN3(d2)

        # d2 = self.Up2(d3)
        # e1 = self.Att2(g=d2, x=e1)
        # d2 = torch.cat((e1, d2), dim=1)
        # d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out


# For nested 3 channels are required

class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        return x


# Nested Unet

class NestedUNet_3(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet_3, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16] * 2

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        # x4_0 = self.conv4_0(self.pool(x3_0))
        # x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_3)
        return output

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

# Dictioary Unet
# if required for getting the filters and model parameters for each step

class ConvolutionBlock(nn.Module):
    """Convolution block"""

    def __init__(self, in_filters, out_filters, kernel_size=3, batchnorm=True, last_active=F.relu):
        super(ConvolutionBlock, self).__init__()

        self.bn = batchnorm
        self.last_active = last_active
        self.c1 = nn.Conv2d(in_filters, out_filters, kernel_size, padding=1)
        self.b1 = nn.BatchNorm2d(out_filters)
        self.c2 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.b2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.c1(x)
        if self.bn:
            x = self.b1(x)
        x = F.relu(x)
        x = self.c2(x)
        if self.bn:
            x = self.b2(x)
        x = self.last_active(x)
        return x


class ContractiveBlock(nn.Module):
    """Deconvuling Block"""

    def __init__(self, in_filters, out_filters, conv_kern=3, pool_kern=2, dropout=0.5, batchnorm=True):
        super(ContractiveBlock, self).__init__()
        self.c1 = ConvolutionBlock(in_filters=in_filters, out_filters=out_filters, kernel_size=conv_kern,
                                   batchnorm=batchnorm)
        self.p1 = nn.MaxPool2d(kernel_size=pool_kern, ceil_mode=True)
        self.d1 = nn.Dropout2d(dropout)

    def forward(self, x):
        c = self.c1(x)
        return c, self.d1(self.p1(c))


class ExpansiveBlock(nn.Module):
    """Upconvole Block"""

    def __init__(self, in_filters1, in_filters2, out_filters, tr_kern=3, conv_kern=3, stride=2, dropout=0.5):
        super(ExpansiveBlock, self).__init__()
        self.t1 = nn.ConvTranspose2d(in_filters1, out_filters, tr_kern, stride=2, padding=1, output_padding=1)
        self.d1 = nn.Dropout(dropout)
        self.c1 = ConvolutionBlock(out_filters + in_filters2, out_filters, conv_kern)

    def forward(self, x, contractive_x):
        x_ups = self.t1(x)
        x_concat = torch.cat([x_ups, contractive_x], 1)
        x_fin = self.c1(self.d1(x_concat))
        return x_fin


class Unet_dict(nn.Module):
    """Unet which operates with filters dictionary values"""

    def __init__(self, n_labels, n_filters=32, p_dropout=0.5, batchnorm=True):
        super(Unet_dict, self).__init__()
        filters_dict = {}
        filt_pair = [3, n_filters]

        for i in range(4):
            self.add_module('contractive_' + str(i), ContractiveBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm))
            filters_dict['contractive_' + str(i)] = (filt_pair[0], filt_pair[1])
            filt_pair[0] = filt_pair[1]
            filt_pair[1] = filt_pair[1] * 2

        self.bottleneck = ConvolutionBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm)
        filters_dict['bottleneck'] = (filt_pair[0], filt_pair[1])

        for i in reversed(range(4)):
            self.add_module('expansive_' + str(i),
                            ExpansiveBlock(filt_pair[1], filters_dict['contractive_' + str(i)][1], filt_pair[0]))
            filters_dict['expansive_' + str(i)] = (filt_pair[1], filt_pair[0])
            filt_pair[1] = filt_pair[0]
            filt_pair[0] = filt_pair[0] // 2

        self.output = nn.Conv2d(filt_pair[1], n_labels, kernel_size=1)
        filters_dict['output'] = (filt_pair[1], n_labels)
        self.filters_dict = filters_dict

    # final_forward
    def forward(self, x):
        c00, c0 = self.contractive_0(x)
        c11, c1 = self.contractive_1(c0)
        c22, c2 = self.contractive_2(c1)
        c33, c3 = self.contractive_3(c2)
        bottle = self.bottleneck(c3)
        u3 = F.relu(self.expansive_3(bottle, c33))
        u2 = F.relu(self.expansive_2(u3, c22))
        u1 = F.relu(self.expansive_1(u2, c11))
        u0 = F.relu(self.expansive_0(u1, c00))
        return F.softmax(self.output(u0), dim=1)


class VGG_Block(nn.Module): # 定义一个为fcn设计的vgg16网络
    def __init__(self,in_channels, out_channels, padding=1 , stride=1,kernel_size=3,bias = False, is_dropout = True,**_):
        super(VGG_Block, self).__init__()
        block = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=padding,stride=stride,kernel_size=kernel_size,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,bias=bias,stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        ]
        if is_dropout:
            block.append(nn.Dropout(0.5))

        self.layer = nn.Sequential(*block)

    def forward(self, x):
        output = self.layer(x)
        return output

class Fcn_8s(nn.Module):
    def __init__(self,im_ch, num_class, **_):
        super(Fcn_8s,self).__init__()
        # 思路：把vgg的卷积部分分成了5个block，区分依据：pool层
        # self.pool1 = VGG_Block(im_ch, 64, is_dropout=False)
        self.pool2 = VGG_Block(im_ch, 128)
        self.pool3 = VGG_Block(im_ch, 128)
        self.pool4 = VGG_Block(128,256)
        self.pool5 = VGG_Block(256, 512)

        # conv6 和 conv7
        conv6 = nn.Conv2d(in_channels=512, out_channels=4096, padding=2, stride=1,kernel_size=5,bias=False)
        bn6 = nn.BatchNorm2d(4096)
        relu6 = nn.ReLU()
        dropout6 = nn.Dropout(0.5)
        conv7 = nn.Conv2d(in_channels=4096, out_channels=4096,stride=1,kernel_size=1,bias=False)
        bn7 = nn.BatchNorm2d(4096)
        relu7 = nn.ReLU()
        dropout7 = nn.Dropout(0.5)
        output = nn.Conv2d(4096, num_class, kernel_size=1)

        # 把这两个conv合并,得到heatmap：
        self.to_heatmap = nn.Sequential(*[
            conv6, bn6, relu6, dropout6, conv7, bn7, relu7, dropout7, output
        ])

        # 用于调整pool3 和 pool4 层的深度，以便做加法
        self.pool3_to_numclass = nn.Conv2d(128, num_class, kernel_size=1)
        self.pool4_to_numclass = nn.Conv2d(256, num_class, kernel_size=1)

        # 用于对conv7上采样，并与调整后的pool4相加
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        # 用于对up1继续上采样，并与调整后的pool3相加
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # 8倍上采样
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.sig = nn.Sigmoid()






    def forward(self, x):
        # pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool3 = self.pool3(x)
        pool3_to_numclass = self.pool3_to_numclass(pool3)
        pool4 = self.pool4(pool3)
        pool4_to_numclass = self.pool4_to_numclass(pool4)
        pool5 = self.pool5(pool4)
        to_heatmap = self.to_heatmap(pool5)
        print(to_heatmap.shape)
        up1 = self.up1(to_heatmap)
        up2 = self.up2(up1 + pool4_to_numclass)
        up3 = self.up3(up2 + pool3_to_numclass)
        # out = F.sigmoid(up3)
        # out[out > 0.5] = 1
        # out[out < 0.5] = 0
        return up3


# U net 的相关代码
class EncoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=2, use_res=False):
        super(EncoderBlock, self).__init__()

        self.use_res = use_res

        self.conv = [nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
            norm(ch_out),
            activation(),
        )]

        for i in range(1, depth):
            self.conv.append(nn.Sequential(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
                                           norm(ch_out),
                                           nn.Sequential() if use_res and i == depth - 1 else activation()
                                           ))
        self.conv = nn.Sequential(*self.conv)
        if use_res:
            self.conv1x1 = nn.Conv2d(ch_in, ch_out, 1)

    def forward(self, x):
        if self.use_res:
            residual = self.conv1x1(x)

        x = self.conv(x)

        if self.use_res:
            x += residual
            x = F.relu(x)

        return x


class DecoderBlock(nn.Module):
    """
    Interpolate
    """

    def __init__(self, ch_in, ch_out, use_deconv=False):
        super(DecoderBlock, self).__init__()
        if use_deconv:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                norm(ch_out),
                activation()
            )

    def forward(self, x):
        return self.up(x)

# 少一层的Unet
class Unet(nn.Module):
    def __init__(self, img_ch=1, num_classes=3, depth=3):
        super(Unet, self).__init__()

        chs = [64, 128, 256, 512, 1024]
        # chs = [44, 88, 176, 352, 704]
        # chs = [36, 72, 144, 288, 360]
        # chs = [16, 32, 64, 128, 256]
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.enc1 = EncoderBlock(img_ch, chs[0], depth=depth)
        self.enc2 = EncoderBlock(img_ch, chs[1], depth=depth)
        self.enc3 = EncoderBlock(chs[1], chs[2], depth=depth)
        self.enc4 = EncoderBlock(chs[2], chs[3], depth=depth)
        self.enc5 = EncoderBlock(chs[3], chs[4], depth=depth)

        self.dec4 = DecoderBlock(chs[4], chs[3])
        self.decconv4 = EncoderBlock(chs[3] * 2, chs[3])

        self.dec3 = DecoderBlock(chs[3], chs[2])
        self.decconv3 = EncoderBlock(chs[2] * 2, chs[2])

        self.dec2 = DecoderBlock(chs[2], chs[1])
        self.decconv2 = EncoderBlock(chs[1] * 2, chs[1])

        # self.dec1 = DecoderBlock(chs[1], chs[0])
        # self.decconv1 = EncoderBlock(chs[0] * 2, chs[0])

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(chs[1], chs[0], kernel_size=3, padding=1, bias=False),
            norm(chs[0]),
            activation(),
            nn.Conv2d(chs[0], num_classes, kernel_size=1, padding=0, bias=False)
        )

        # initialize_weights(self)

    def forward(self, x):
        # encoding path
        # x1 = self.enc1(x)
        #
        # x2 = self.maxpool(x1)
        x2 = self.enc2(x)

        x3 = self.maxpool(x2)
        x3 = self.enc3(x3)

        x4 = self.maxpool(x3)
        x4 = self.enc4(x4)

        x5 = self.maxpool(x4)
        x5 = self.enc5(x5)

        # decoding + concat path
        d4 = self.dec4(x5)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.decconv4(d4)

        d3 = self.dec3(d4)

        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.decconv3(d3)
        d2 = self.dec2(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.decconv2(d2)


        d1 = self.conv_1x1(d2)
        return d1




# import torch
import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
class VGG_Block(nn.Module): # 定义一个为fcn设计的vgg16网络
    def __init__(self,in_channels, out_channels, padding=1 , stride=1,kernel_size=3,bias = False, is_dropout = True,**_):
        super(VGG_Block, self).__init__()
        block = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=padding,stride=stride,kernel_size=kernel_size,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,bias=bias,stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        ]
        if is_dropout:
            block.append(nn.Dropout(0.5))

        self.layer = nn.Sequential(*block)

    def forward(self, x):
        output = self.layer(x)
        return output

class fcn(nn.Module):
    def __init__(self,im_ch=1, num_class=2, **_):
        super(fcn,self).__init__()
        # 思路：把vgg的卷积部分分成了5个block，区分依据：pool层
        self.pool1 = VGG_Block(im_ch, 64, is_dropout=False)
        self.pool2 = VGG_Block(im_ch, 128)
        self.pool3 = VGG_Block(128, 256)
        self.pool4 = VGG_Block(256,512)
        self.pool5 = VGG_Block(512, 512)

        # conv6 和 conv7
        conv6 = nn.Conv2d(in_channels=512, out_channels=4096, padding=2, stride=1,kernel_size=5,bias=False)
        bn6 = nn.BatchNorm2d(4096)
        relu6 = nn.ReLU()
        dropout6 = nn.Dropout(0.5)
        conv7 = nn.Conv2d(in_channels=4096, out_channels=4096,stride=1,kernel_size=1,bias=False)
        bn7 = nn.BatchNorm2d(4096)
        relu7 = nn.ReLU()
        dropout7 = nn.Dropout(0.5)
        output = nn.Conv2d(4096, num_class, kernel_size=1)

        # 把这两个conv合并,得到heatmap：
        self.to_heatmap = nn.Sequential(*[
            conv6, bn6, relu6, dropout6, conv7, bn7, relu7, dropout7, output
        ])

        # 用于调整pool3 和 pool4 层的深度，以便做加法
        self.pool3_to_numclass = nn.Conv2d(256, num_class, kernel_size=1)
        self.pool4_to_numclass = nn.Conv2d(512, 3, kernel_size=1)

        # 用于对conv7上采样，并与调整后的pool4相加
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        # 用于对up1继续上采样，并与调整后的pool3相加
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # 8倍上采样
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear')
        # self.sig = nn.Sigmoid()






    def forward(self, x):
        # pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool3 = self.pool3(pool2)
        pool3_to_numclass = self.pool3_to_numclass(pool3)
        pool4 = self.pool4(pool3)
        pool4_to_numclass = self.pool4_to_numclass(pool4)
        pool5 = self.pool5(pool4)
        to_heatmap = self.to_heatmap(pool5)

        up1 = self.up1(to_heatmap)
        up2 = self.up2(up1 + pool4_to_numclass)
        up3 = self.up3(up2 + pool3_to_numclass)
        # out = F.sigmoid(up3)
        # out[out > 0.5] = 1
        # out[out < 0.5] = 0
        return up3


if __name__ == '__main__':
    net = NestedUNet(in_ch=3, out_ch=2)
    a = torch.rand((1, 3, 200, 200))
    b = net(a)
    print(b.shape)

