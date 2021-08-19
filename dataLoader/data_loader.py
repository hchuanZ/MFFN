# 约定：默认都给projection map， net用不用由net决定，然后有多个模态的projection map嘛，具体咋个用，应该要留好接口撒(不用了，传入不同的projection路径就好了)
# 要能都同时适应两个分割任务的需求
# 最好还是要能够overlip吧，overlip然后看看指标有没有提升？
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from natsort import natsorted
import imageio
import torch.nn.functional as F
from helper.utils import *
import warnings
warnings.filterwarnings("ignore")

class DataSet(Dataset):
    def __init__(self, oct_path, octa_path, gt_path, pro_path, isTrain='train', sampling_rate=0.5, task='RV',  is_norm=False):
        data_name_list = natsorted(os.listdir(oct_path))
        gt_name_list = natsorted(os.listdir(gt_path))
        if isTrain == 'train':
            oct_name_list = data_name_list[:180]
            octa_name_list = data_name_list[:180]
            label_name_list = gt_name_list[:180]
            pro_name_list = gt_name_list[:180]

        elif isTrain == 'val':
            oct_name_list = data_name_list[180:200]
            octa_name_list = data_name_list[180:200]
            label_name_list = gt_name_list[180:200]
            pro_name_list = gt_name_list[180:200]

        else:
            oct_name_list = data_name_list[200:]
            octa_name_list = data_name_list[200:]
            label_name_list = gt_name_list[200:]
            pro_name_list = gt_name_list[200:]

        self.oct_path = [os.path.join(oct_path, name) for name in oct_name_list]
        self.octa_path = [os.path.join(octa_path, name) for name in octa_name_list]
        self.label_path = [os.path.join(gt_path, name) for name in label_name_list]
        self.pro_path = [os.path.join(pro_path, name) for name in pro_name_list]
        self.isTrain = isTrain
        self.samp = sampling_rate
        self.task = task
        # self.crop_size = crop_size
        # self.use_pro = use_pro
        self.is_norm = is_norm

    def __getitem__(self, item):
        oct_name = self.oct_path[item]
        octa_name = self.octa_path[item]
        gt_name = self.label_path[item]
        pro_name = self.pro_path[item]

        oct_data = np.load(oct_name)[:, :, 160:480]
        octa_data = np.load(octa_name)[:, :, 160:480]

        if self.task == 'FAZ':
            oct_data = crop_center_3d(oct_data, 200, 200, 320)
            octa_data = crop_center_3d(octa_data, 200, 200, 320)

        # 归一化
        if self.is_norm:
            # octa_data = z_score(octa_data)
            # oct_data = z_score(oct_data)
            oct_data = z_score(oct_data)
            octa_data = z_score(octa_data)
        gt_data = np.array(imageio.imread(gt_name))
        pro_data = np.array(imageio.imread(pro_name))
        if self.task == 'FAZ':
            gt_data = crop_center(gt_data, 200, 200)
            pro_data = crop_center(pro_data, 200, 200)
        if self.task == 'RV':
            gt_data[gt_data == 100] = 0
            palette = [[0], [255]]
        else:
            gt_data[gt_data == 255] = 0
            palette = [[0], [100]]
        gt_data = np.expand_dims(gt_data, axis=-1)
        pro_data = np.expand_dims(pro_data, 0)
        pro_data = pro_data / 255
        gt_data = targer_to_onehot(gt_data, palette=palette)
        gt_data = np.transpose(gt_data, (2, 0, 1))
        gt = gt_data[1]  # 只留下前景
        gt_slices = []
        octa_slice = []
        if self.task == 'RV':
            for i in range(1, 5):
                for j in range(1, 5):
                    gt_patch = gt[100 * (i - 1):100 * i, 100 * (i - 1):100 * i]
                    gt_slices.append(gt_patch)

                    octa_patch = octa_data[100 * (i - 1):100 * i, 100 * (i - 1):100 * i, :]
                    octa_slice.append(octa_patch)
        else:
            for i in range(1, 3):
                for j in range(1, 3):
                    gt_patch = gt[100 * (i - 1):100 * i, 100 * (i - 1):100 * i]
                    gt_slices.append(gt_patch)

                    octa_patch = octa_data[100 * (i - 1):100 * i, 100 * (i - 1):100 * i, :]
                    octa_slice.append(octa_patch)
        gt_slices = np.array(gt_slices)
        octa_slice = np.array(octa_slice)
        gt = np.expand_dims(gt, axis=0)

        octa_data = torch.from_numpy(octa_data)
        oct_data = torch.from_numpy(oct_data)
        gt_slices = torch.from_numpy(gt_slices)
        gt = torch.from_numpy(gt)

        data = [oct_data, octa_data]
        data = torch.stack(data, dim=0).float()
        # print(data.shape)
        data = F.upsample(data, scale_factor=(1, self.samp), mode='bilinear')
        octa_slice = torch.from_numpy(octa_slice).float()
        # print(octa_slice.shape)
        octa_slice = F.upsample(octa_slice, scale_factor=(1, self.samp), mode='bilinear')
        pro_data = torch.from_numpy(pro_data).float()
        return data, octa_slice, gt, gt_slices, pro_data

    def __len__(self):
        return len(self.octa_path)

def getBiDataLoder(path, isTrain='train',  is_norm=True, sampling_rate=0.5, task='RV', use_pro=True, batch_size = 1, is_shuffle=True, subset='6m'):
    data = DataSet(path[0], path[1], path[2], path[3], isTrain,  sampling_rate, task,  is_norm)
    dataLoader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=is_shuffle,
        num_workers=8,
        pin_memory=True
    )
    return dataLoader, len(data)