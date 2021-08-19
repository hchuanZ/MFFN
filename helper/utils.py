import torch
import torch.nn as nn
import numpy as np
import torch
import nibabel as nib

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def crop_center_3d(img, cropx, cropy, cropz):
    y, x, z = img.shape
    startx = x // 2 - (cropx//2)
    starty = y//2 - (cropy // 2)
    startz = z // 2  - (cropz// 2)

    return img[starty:starty+cropy, startx:startx+cropx, startz:startz+cropz]

def nib_read(path):
    img = np.array(nib.load(path).get_fdata())
    return img

def init_random_(full=[400,400,640], crop=[112,112,0]): # 这里为x，y，z三个尺度生产random val
    # if ver == 2:
    #     startx = np.random.randint(0, 288)
    #     starty = np.random.randint(0, 288)
    #     startz = np.random.randint(0, 140)
    # elif ver == 1:
    #     startx = np.random.randint(0, 300)
    #     starty = np.random.randint(0, 300)
    #     startz = np.random.randint(0, 140)
    startx = np.random.randint(0, full[0]-crop[0])
    starty = np.random.randint(0, full[1]-crop[1])
    startz = np.random.randint(0, full[2]-crop[2])
    return startx, starty, startz


def init_random(ver): # 这里为x，y，z三个尺度生产random val
    if ver == 2:
        startx = np.random.randint(0, 288)
        starty = np.random.randint(0, 288)
        startz = np.random.randint(0, 140)
    elif ver == 1:
        startx = np.random.randint(0, 300)
        starty = np.random.randint(0, 300)
        startz = np.random.randint(0, 140)
    return startx, starty, startz

def crop(img, startx, starty, startz = None, crop_size=100):
    overlip = crop_size
    if startz == None:
        # if ver == 2:
        # 没有传入 startz，应该是个二维的图，即gt
        #     img = img[startx:startx+112, starty:starty+112]
        # elif ver == 1:
        img = img[startx:startx + overlip, starty:starty + overlip]
    elif startz < 0:
        # 是三维的输入，但是Z轴不需要裁剪
        # if ver == 2:
        img = img[startx:startx+overlip, starty:starty+overlip, :]
        # elif ver == 1:
        #     img = img[startx:startx+100, starty:starty+100, :]

    else:
        # if ver == 2:
        img = img[startx:startx+overlip, starty:starty+overlip, startz:startz+overlip]
        # elif ver == 1:
        #     img = img[startx:startx + 100, starty:starty + 100, startz:startz + 100]
    return img

def getListPathFromTxt(path): # 从txt中读取数据
    f1 = open(path)
    lines = f1.readlines()
    out = []
    for i in lines:
        out.append(i[:-1])
    return out


def targer_to_onehot(target, palette): # palette是颜色表，根据自己的数据集来分类好
    seg_map = []
    for color in palette:
        equals = np.equal(target, color)# 找到target中和当前分类的颜色相同的
        class_map = np.all(equals,axis=-1)
        seg_map.append(class_map)
    seg_map = np.stack(seg_map, axis=-1).astype(np.float32)
    return seg_map

def z_score(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std

def ct_preprocess(img, yu_val_low, yu_val_high):  # yu_val_low表示阈值的下界
    img[img > yu_val_high] = yu_val_high
    img[img < yu_val_low] = yu_val_low
    img = (img - yu_val_low) / (yu_val_high - yu_val_low)
    return img

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N

def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    # 如果不加dim=1，那么得到的是这个batch的多张pic tp的和，加了，则得到一个list，len为batch_size
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N

class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation=None, reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []
        for i in range(0, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
            # print('dicoddof',diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
        mean_dice = sum(class_dice) / len(class_dice)

        #  不用平均dice了，用加权的
        # weight_dice = class_dice[0] * 0.3 + class_dice[1] * 0.7
        return 1 - mean_dice



if __name__ == '__main__':
    a = np.random.randint(-300,400,[3,3])
    print(a)
    b = ct_preprocess(a, -120, 240)
    print(b)
     # right


