import torch.nn as nn
import torch

activation = nn.ReLU
gnorm = nn.GroupNorm
norm = nn.BatchNorm2d

class PLM_res(nn.Module):
    def __init__(self, in_channels, out_channels, pooling_size, k_size=3, pool_type='max', res_rate=0.1):
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
            # nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            # gnorm(out_channels//16 ,out_channels),
            # nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, pooling_size), stride=(1, 1, pooling_size)),
            gnorm(out_channels//16 ,out_channels),
            nn.ReLU(inplace=True),
        )
        self.res_rate = res_rate


    def forward(self, x):
        # print(x.shape)
        # return self.plm(x) + 0.5 * self.res(x)
        return self.plm(x) + self.res_rate * self.res(x)

class skip(nn.Module):
    def __init__(self, in_channels, out_channels, cube_size, k_size):
        super(skip, self).__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=k_size, padding=k_size // 2),
            gnorm(out_channels//16 ,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, cube_size), stride=(1,1,cube_size))
        )
    def forward(self,x):
        return self.skip_conv(x)
