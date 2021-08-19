import os
# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
from helper.metrics import jaccardv2
from helper.utils import *
from helper.metrics import balance_acc

import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import time
subset = '6m'
dataroot = '/media/upup/zhc/dataset/OCTA500/full_npy'
rootpath = '/media/upup/zhc/code/BI_IPN_workspace/log/pth'
root = '/media/upup/zhc/code/BI_IPN_workspace'
if subset == '6m':
    from dataLoader.data_loader import getBiDataLoder
    path = [
        os.path.join(dataroot, 'OCT_6m'),
        os.path.join(dataroot, 'OCTA_6m'),
        os.path.join(dataroot, r'gt/6m'),
        os.path.join(dataroot, 'gt/projection/OCTA_ILM_OPL')
    ]
else:
    from dataLoader.data_loader_3m import getBiDataLoder
    path = [
        os.path.join(dataroot, 'OCT_3m'),
        os.path.join(dataroot, 'OCTA_3m'),
        os.path.join(dataroot, r'gt/3m'),
        os.path.join(dataroot, 'gt/projection/3m')
    ]

sampling_rate = 0.25
task = 'FAZ'
is_norm = False
number = 72

save_img_path = os.path.join(root, 'OUTPUT/number'+str(number))
pth_save_name = './log/pth/checkpoint_Bi_IPN_FAZ_43_star.pth'
def test_main():
    pth = pth_save_name
    model = torch.load(pth).cuda()

    Loader, nums = getBiDataLoder(path=path, isTrain='test', task=task, is_shuffle=False, sampling_rate=sampling_rate,
                                 subset=subset)
    test(Loader, nums, model)
    countparam(model)

def test(Loader, nums, net):
    Dice = []
    JACC = []
    BACC = []
    flag = 1
    with torch.no_grad():
        for _, (data, gt) in enumerate(Loader):
        # for _, (data, octa_slice, gt, gt_slices, pro_data) in enumerate(Loader):
            data = data.cuda()
            # octa_slice = octa_slice.cuda()
            gt = gt.cuda()
            # gt_slices = gt_slices.cuda()
            # pro_data = pro_data.cuda()
            # pred, _, _ = net(data, octa_slice, pro_data)
            pred = net(data)
            # 训练出来的是一个416 * 416的图，我们支取中间的400*400
            # pred = TTF.center_crop(pred, (400, 400))
            pred = torch.sigmoid(pred)
            pred[pred > 0.5] = 1
            pred[pred < 0.5] = 0
            if subset == '6m':
                pad_size = 100
            else:
                pad_size = 52
            if task == 'FAZ':
                gt = torch.nn.functional.pad(gt, pad=(pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
                pred = torch.nn.functional.pad(pred, pad=(pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            dice = diceCoeffv2(pred[:, 1:2, :], gt[:, 0:1, :], activation=None).cpu()
            # dice = diceCoeffv2(pred[:, 0:1, :], gt[:, 0:1, :], activation=None).cpu()
            Dice.append(dice)
            # jac
            jacc = jaccardv2(pred[:, 1:2, :], gt[:, 0:1, :]).cpu()
            # jacc = jaccardv2(pred[:, 0:1, :], gt[:, 0:1, :]).cpu()
            JACC.append(jacc)

            # bacc = balance_acc(pred[:, 0:1, :], gt[:, 0:1, :]).cpu()
            bacc = balance_acc(pred[:, 1:2, :], gt[:, 0:1, :]).cpu()
            BACC.append(bacc)

            pred_img = pred[0][1]
            pred_img = pred_img.detach().cpu().numpy()
            pred_img = pred_img * 255
            save_img(save_img_path, pred_img, 200 + flag, dice)
            print('---val_num----: ' + str(flag))
            print(dice)
            flag += 1

        mean_dice = sum(Dice) / nums
        dice_std = np.array(Dice).std()
        mean_jacc = sum(JACC) / nums
        jacc_std = np.array(JACC).std()
        mean_bacc = sum(BACC) / nums
        bacc_std = np.array(BACC).std()
        print('TEST ______________________')
        print('mean_dice : %.4f, std : %.4f' % (mean_dice, dice_std))
        print('mean_jacc : %.4f, std : %.4f' % (mean_jacc, jacc_std))
        print('mean_bacc : %.4f, std : %.4f' % (mean_bacc, bacc_std))
        print('TEST ______________________')

def save_img(pth, img, number, dice):
    im = Image.fromarray(img)
    path = os.path.join(pth, str(number)+'_res_BI_IPN'+task+'_'+str(dice) + '.jpeg')
    im.convert('1').save(path)
def countparam(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
if __name__ == '__main__':
    test_main()