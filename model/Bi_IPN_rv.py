
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.nets_tools import NestedUNet_3 as unetplusplus
from model.nets_tools import AttU_Net_3 as attunet
from model.nets_tools import R2U_Net_3 as r2unet
from model.nets_tools import Fcn_8s as fcn

activation = nn.ReLU
gnorm = nn.GroupNorm
norm = nn.BatchNorm2d


class PLM_res(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_size, k_size=3, pool_type='max'):
        super(PLM_res, self).__init__()
        # print(pooling_size)
        if pool_type=='max':
            self.plm = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=k_size, padding=k_size // 2),
                gnorm(out_channels//16 ,out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=k_size, padding=k_size // 2),
                gnorm(out_channels//16 ,out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=[1, 1, pooling_size])
            )
        else:
            self.plm = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=k_size, padding=k_size // 2),
                gnorm(out_channels//16 ,out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=k_size, padding=k_size // 2),
                gnorm(out_channels//16 ,out_channels),
                nn.ReLU(inplace=True),
                nn.AvgPool3d(kernel_size=[1, 1, pooling_size])
            )
        
        self.res = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            gnorm(out_channels//16 ,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, pooling_size), stride=(1, 1, pooling_size)),
            gnorm(out_channels//16 ,out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        # print(x.shape)
        return self.plm(x) + 0.5 * self.res(x)


class PLM(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_size, k_size=3, pool_type='max'):
        super(PLM, self).__init__()
        # print(pooling_size)
        if pool_type=='max':
            self.plm = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=k_size, padding=k_size // 2),
                gnorm(out_channels//16 ,out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=k_size, padding=k_size // 2),
                gnorm(out_channels//16 ,out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=[1, 1, pooling_size])
            )
        else:
            self.plm = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=k_size, padding=k_size // 2),
                gnorm(out_channels//16 ,out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=k_size, padding=k_size // 2),
                gnorm(out_channels//16 ,out_channels),
                nn.ReLU(inplace=True),
                nn.AvgPool3d(kernel_size=[1, 1, pooling_size])
            )


    def forward(self, x):
        # print(x.shape)
        return self.plm(x)

class rough_IPN(nn.Module):
    def __init__(self, in_ch=2, in_cube=80, plm_nums=3, use_res = False):
        super(rough_IPN, self).__init__()

        pool_size_list = [
            [0], [0],  # 这两个是拿来凑数的
            [10, 8],
            [5, 4, 4],
            [5, 4, 2, 2],
            [5, 2, 2, 2, 2]
        ]
        chs_list = [
            [0], [0],
            [in_ch, 32, 64],
            [in_ch, 16, 32, 64],
            [in_ch, 16, 32, 32, 64],
            [in_ch, 16, 32, 32, 32, 64]
        ]
        k_size_list = [
            [0],[0],
            [3, 3],
            [3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3, 3]
        ]
        pool_type_list = [
            [0],[0],
            ['max', 'max'],
            ['max', 'max', 'max'],
            ['max', 'max', 'max', 'max'],
            ['max', 'max', 'max', 'max', 'max']
        ]
        chs = chs_list[plm_nums]
        pool_size = pool_size_list[plm_nums]
        k_size = k_size_list[plm_nums]
        pool_type = pool_type_list[plm_nums]
        if use_res :
            plms = [PLM_res(in_channels=chs[i], out_channels=chs[i+1], pooling_size=pool_size[i], k_size=k_size[i], pool_type=pool_type[i]) for i in range(len(pool_size))]
        else:
            plms = [PLM(in_channels=chs[i], out_channels=chs[i+1], pooling_size=pool_size[i], k_size=k_size[i], pool_type=pool_type[i]) for i in range(len(pool_size))]
        self.plms = nn.Sequential(*plms)
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channels=chs[-1], out_channels=chs[1], kernel_size=1),
            gnorm(2, chs[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=chs[1], out_channels=1, kernel_size=1)
        ])

    def forward(self, x):
        x = self.plms(x)
        x = torch.squeeze(x, dim=-1)
        x = self.conv(x)
        return x

class fine_IPN(nn.Module):
    def __init__(self, in_ch=2, in_cube=80, plm_nums=3, slices=16, use_res=False):
        super(fine_IPN, self).__init__()

        pool_size_list = [
            [0], [0],  # 这两个是拿来凑数的
            [10, 8],
            [5, 4, 4],
            [5, 4, 2, 2],
            [5, 2, 2, 2, 2]
        ]
        chs_list = [
            [0], [0],
            [in_ch, 128, 512],
            [in_ch, 128, 256, 512],
            [in_ch, 64, 128, 256, 512],
            [in_ch, 64, 128, 256, 256, 512]
        ]
        k_size_list = [
            [0], [0],
            [3, 3],
            [3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, 3, 3]
        ]
        pool_type_list = [
            [0], [0],
            ['max', 'max'],
            ['max', 'max', 'max'],
            ['max', 'max', 'max', 'max'],
            ['max', 'max', 'max', 'max', 'max']
        ]
        chs = chs_list[plm_nums]
        pool_size = pool_size_list[plm_nums]
        k_size = k_size_list[plm_nums]
        pool_type = pool_type_list[plm_nums]
        if use_res:
            plms = [PLM_res(in_channels=chs[i], out_channels=chs[i+1], pooling_size=pool_size[i], k_size=k_size[i], pool_type=pool_type[i]) for i in range(len(pool_size))]
        else:
            plms = [PLM(in_channels=chs[i], out_channels=chs[i+1], pooling_size=pool_size[i], k_size=k_size[i], pool_type=pool_type[i]) for i in range(len(pool_size))]
        self.plms = nn.Sequential(*plms)
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channels=chs[-1], out_channels=128, kernel_size=1),
            gnorm(128//16 ,128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=slices, kernel_size=1)
        ])

    def forward(self, x):
        x = self.plms(x)
        x = torch.squeeze(x, dim=-1)
        x = self.conv(x)
        return x

class Bi_IPN(nn.Module):
    def __init__(self, num_class=2, in_cube=80, use_projection=True, fusion_net='unet', task='RV', plm_nums=3, use_res=False):
        super(Bi_IPN, self).__init__()
        if task == 'RV':
            slice = 16
        else:
            slice = 4
        self.rough = rough_IPN(in_cube=in_cube, plm_nums=plm_nums, use_res=use_res)
        self.fine = fine_IPN(in_cube=in_cube, plm_nums=plm_nums, slices=slice, in_ch=slice, use_res=use_res)
        if use_projection:  
            im_ch = 3
        else:
            im_ch = 2
        if fusion_net == 'ushaped':
        	self.fusion = unet(img_ch=im_ch, num_classes=num_class)
        elif fusion_net == 'fcn':
            self.fusion = fcn(im_ch=im_ch, num_class=num_class)
        elif fusion_net == 'r2unet':
            self.fusion = r2unet(img_ch=im_ch, output_ch=num_class)
        elif fusion_net == 'unet++':
            self.fusion = unetplusplus(in_ch=im_ch, out_ch=num_class)
        elif fusion_net == 'attunet':
            self.fusion = attunet(img_ch=im_ch, output_ch=num_class)

        # else:
        #     self.unet = Baseline(img_ch=2, num_classes=num_class)
        # self.use_projection = use_projection
        # self.pad = nn.ReflectionPad2d(8)
        self.slice = slice
        self.task = task

    def forward(self, data, data_slice, projection_map):
        # print(data_slice.shape)
        rough_mask = self.rough(data)
        fine_mask = self.fine(data_slice)
        y1 = rough_mask  # 粗分割结果，拿去做损失
        y2 = fine_mask  # 细分割结果，也拿去做损失

        # 矩阵拼接工作
        # 首先把多个100*100的取出来
        lists = []
        for i in range(self.slice):
            mask = fine_mask[:, i, :]
            lists.append(mask)
        if self.task == 'RV':
            m1 = torch.cat((lists[0], lists[1], lists[2], lists[3]), 2)
            m2 = torch.cat((lists[4], lists[5], lists[6], lists[7]), 2)
            m3 = torch.cat((lists[8], lists[9], lists[10], lists[11]), 2)
            m4 = torch.cat((lists[12], lists[13], lists[14], lists[15]), 2)
            mask_full = torch.cat((m1, m2, m3, m4), 1)

        else:
            m1 = torch.cat((lists[0], lists[1]), 2)
            m2 = torch.cat((lists[2], lists[3]), 2)
            mask_full = torch.cat((m1, m2), 1)  # 这就是细分割的结果拼接出来的图

        mask_full = torch.unsqueeze(mask_full, dim=1)

        x = torch.cat((rough_mask, mask_full, projection_map), 1).float()
        y = self.fusion(x)  # 给融合网络去继续train

        return y, y1, y2





# ushaped 相关的代码
class EncoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=2, use_res=False):
        super(EncoderBlock, self).__init__()

        self.use_res = use_res

        self.conv = [nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
            gnorm(ch_out//16, ch_out),
            activation(inplace=True),
        )]

        for i in range(1, depth):
            self.conv.append(nn.Sequential(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
                                           gnorm(ch_out//16, ch_out),
                                           nn.Sequential() if use_res and i == depth - 1 else activation(inplace=True)
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
                gnorm(ch_out//16, ch_out),
                activation()
            )

    def forward(self, x):
        return self.up(x)


class unet(nn.Module):
    def __init__(self, img_ch=1, num_classes=3, depth=3):
        super(unet, self).__init__()

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
            gnorm(chs[0]//16, chs[0]),
            activation(inplace=True),
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


# unet++ code




# attention unet code




# simple fusion




# se-unet
