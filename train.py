import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from helper.utils import *
from helper.pytorchtools import *
from dataLoader.data_loader_crop import getBiDataLoder
import time
import warnings
import math
from helper.metrics import jaccardv2
from helper.metrics import balance_acc
from PIL import Image
warnings.filterwarnings("ignore")
dataroot = '/media/upup/d6c7e0e6-efe4-4f7b-9f42-bfcc4fe51fee/zhc/dataset/OCTA500/full_npy'
rootpath = '/media/upup/d6c7e0e6-efe4-4f7b-9f42-bfcc4fe51fee/zhc/code/BI_IPN_workspace/log/pth'
root = '/media/upup/d6c7e0e6-efe4-4f7b-9f42-bfcc4fe51fee/zhc/code/BI_IPN_workspace'


epochs = 500
# crop_size = 96
early_stop_patience = 12
per_epoch_val = 1  # 每多少次迭代做一回验证
initial_lr = 0.0001
threshold_lr = 1e-6
weight_decay = 1e-5
sampling_rate = 0.25
lr_scheduler_eps = 1e-3
batch_size = 4
decay_rate = 0.99
loss_name = 'bce'
optimizer_type = 'adam'
net_name = 'Bi_IPN'
is_norm = False
number = 96  # 实验的次数，作为代号,就当是个id
use_pro = True  # 是否使用投影图
# use_aux_loss = False
# aux_rate = 1
task = 'FAZ' # 任务是血管分割还是FAZ分割
fusion_net = 'unet'   # 融合网络的选择
plm_nums = 3    # plm模块的数量
pth_save_name = os.path.join(rootpath, 'checkpoint_'+net_name + '_' + task +'_'+str(number)+'.pth')
save_img_path = os.path.join(root, 'OUTPUT/number'+str(number))
path = [
    os.path.join(dataroot, 'OCT_6m'),
    os.path.join(dataroot, 'OCTA_6m'),
    os.path.join(dataroot, r'gt/6m'),
    os.path.join(dataroot, 'gt/projection/OCTA_ILM_OPL')
]

if net_name == 'Bi_IPN':
    from model.Bi_IPN_full_samp import Bi_IPN as BaseLine



def lr_adjust(optimizer, now_lr, decay_rate):
    if optimizer_type=='adam':
        lr = now_lr * decay_rate
    else:
        lr = math.pow(now_lr, decay_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def countparam(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')


def main():
    net = BaseLine(num_class=2, in_cube=int(320 * sampling_rate), plm_nums=plm_nums,task=task, fusion_net=fusion_net).cuda()
    countparam(net)
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(), lr=initial_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9)

    # 定义损失函数
    if loss_name == 'dice':
        criterion = SoftDiceLoss(num_classes=2).cuda()
    elif loss_name == 'bce':
        criterion = torch.nn.BCELoss().cuda()

    early_stoping = EarlyStopping(early_stop_patience, verbose=True, delta=lr_scheduler_eps,path=pth_save_name)


    train_loader, _ = getBiDataLoder(path=path, crop_size=crop_size, task=task, sampling_rate=sampling_rate,
                                     batch_size=batch_size, is_norm=is_norm)
    val_loader, val_nums = getBiDataLoder(path, isTrain='val', task=task, crop_size=crop_size,
                                          sampling_rate=sampling_rate, is_shuffle=False, is_norm=is_norm)

    train(train_loader, val_loader, net, criterion,
          optimizer, early_stoping, epochs, val_nums, use_pro, initial_lr)

    test_main()

def train(train_loader, val_loader, net, criterion, optimizer, early_stop, epochs, val_nums, use_pro, initial_lr):
    criterion_dice = SoftDiceLoss(num_classes=1).cuda()
    iter = 0
    overlip = crop_size - 100
    train_loss = []
    train_dice = []
    for epoch in range(1, epochs):
        epoch_t = np.array([epoch]) / aux_rate
        epoch_t = torch.from_numpy(epoch_t)
        epoch_time = time.time()
        for nums_batch, (data, octa_slice, gt, gt_slices, pro_data) in enumerate(train_loader):

            # cuda
            data = data.cuda()
            octa_slice = octa_slice.cuda()
            gt = gt.cuda()
            gt_slices = gt_slices.cuda()
            pro_data = pro_data.cuda()

            optimizer.zero_grad()

            y, y1, y2 = net(data, octa_slice, pro_data)
            y = torch.sigmoid(y)
            y1 = torch.sigmoid(y1)
            y2 = torch.sigmoid(y2)
            # print(y.shape)
            # if use_pro:
            #     y = torch.sigmoid(net(volume, projection_map))
            # else:
            #     if net_name == 'denseIPN' or net_name=='denseIPN_2':
            #         y1, y = net(volume)
            #         y1 = torch.sigmoid(y1)
            #         y = torch.sigmoid(y)
            #     else:
            #         y = torch.sigmoid(net(volume))
            # gt_pad = TTF.pad(gt, 8, padding_mode='reflect')
            # loss = criterion(y[:, 1:2, :], gt[:, 0:1, :]) + criterion_dice(y[:, 1:2, :], gt[:, 0:1, :])
            loss = criterion(y[:, 1:2, :], gt[:, 0:1, :])
            #
            # print(gt.shape)
            if use_aux_loss == True:
                if net_name == 'denseIPN' or net_name == 'denseIPN_2':
                    loss_aux1 = criterion(y1[:, 1:2, :], gt[:, 1:2, :])
                    loss = loss + (1 - torch.sigmoid(epoch_t).numpy()[0]) * loss_aux1
                if net_name == 'Bi_IPN' or net_name == 'Bi_IPN_res':
                    loss_aux1 = criterion(y1, gt)
                    loss_aux2 = criterion(y2, gt_slices)
                    loss = loss * (torch.sigmoid(epoch_t).numpy()[0]) + (1 - torch.sigmoid(epoch_t).numpy()[0]) * (
                                loss_aux1 + loss_aux2)

                # loss = loss_aux1 * 0.5 + loss_aux2 * 0.5 + loss
            # y = TTF.center_crop(y, (400, 400))
            # gt = TTF.center_crop(gt, (400 ,400))
            # print(y.shape, gt.shape)
            # y = y.contiguous()
            # gt = gt.contiguous()
            dicecoeff_rv = diceCoeffv2(y[:, 1:2, :], gt[:, 0:1, :], activation=None).cpu().item()
            train_loss.append(loss.item())
            train_dice.append(dicecoeff_rv)
            loss.backward()
            optimizer.step()

            useTime = time.time() - epoch_time

            print('iter : %d, train_loss : %.4f, rv_dice : %.4f'
                  % (iter, loss.item(), dicecoeff_rv), 'useTime : ' + str(useTime))
            iter += 1
        mean_train_loss = sum(train_loss) / len(train_loss)
        mean_train_dice = sum(train_dice) / len(train_dice)
        print('-------')
        print('epoch : ' + str(epoch) + 'train_mean_loss : ' +
              str(mean_train_loss) + 'train_mean_dice : ' + str(mean_train_dice))
        print('-------')

        Dice = []
        JACC = []
        BACC = []
        if epoch % per_epoch_val == 0:
            with torch.no_grad():
                for num_batch, (data, octa_slice, gt, _, pro_data) in enumerate(val_loader):
                    data = data.cuda()
                    octa_slice = octa_slice.cuda()
                    gt = gt.cuda()
                    # gt_slices = gt_slices.cuda()
                    pro_data = pro_data.cuda()

                    pred, _, _ = net(data, octa_slice, pro_data)
                    pred = torch.sigmoid(pred)
                    # pred = TTF.center_crop(pred, (400, 400))
                    pred[pred > 0.5] = 1
                    pred[pred < 0.5] = 0
                    # pred = pred.cpu().numpy()
                    # pred = pred.contiguous()
                    # gt = gt.contiguous()

                    
                    dice = diceCoeffv2(pred[:, 1:2, :], gt[:, 0:1, :], activation=None).cpu()
                    Dice.append(dice)
                    jacc = jaccardv2(pred[:, 1:2, :], gt[:, 0:1, :]).cpu()
                    JACC.append(jacc)
                    bacc = balance_acc(pred[:, 1:2, :], gt[:, 0:1, :]).cpu()
                    BACC.append(bacc)

                    # gt = data[1].cuda()
                    # imgs = data[0][0]
                    # pros = data[2][0]
                    # preds = []
                    # for i in range(16):
                    #     img = imgs[i].cuda()
                    #     pro = pros[i].cuda()
                    #     if use_pro:
                    #         pred = torch.sigmoid(net(img, pro))
                    #     else:
                    #         if net_name == 'denseIPN' or net_name=='denseIPN_2':
                    #             _, pred = net(img)
                    #             # y1 = torch.sigmoid(y1)
                    #             pred = torch.sigmoid(pred)
                    #         else:
                    #             pred = torch.sigmoid(net(img))
                    #     pred[pred > 0.5] = 1
                    #     pred[pred < 0.5] = 0
                    #     pred = pred.cpu().numpy()
                    #     pred = pred[0][1][overlip // 2:100 +
                    #                                    overlip // 2, overlip // 2:100 + overlip // 2]
                    #     preds.append(pred)
                    # img1 = np.concatenate(
                    #     (preds[0], preds[1], preds[2], preds[3]), axis=1)
                    # img2 = np.concatenate(
                    #     (preds[4], preds[5], preds[6], preds[7]), axis=1)
                    # img3 = np.concatenate(
                    #     (preds[8], preds[9], preds[10], preds[11]), axis=1)
                    # img4 = np.concatenate(
                    #     (preds[12], preds[13], preds[14], preds[15]), axis=1)

                    # img = np.concatenate((img1, img2, img3, img4), axis=0)
                    # img = np.expand_dims(img, axis=0)
                    # imgTensor = torch.from_numpy(np.expand_dims(img, axis=0)).cuda()
                    # dice = diceCoeffv2(imgTensor[:, 0:1, :], gt[:, 1:2, :], activation=None).cpu()
                    # Dice.append(dice)

                mean_dice = sum(Dice) / val_nums
                dice_std = np.array(Dice).std()
                mean_jacc = sum(JACC) / val_nums
                jacc_std = np.array(JACC).std()
                mean_bacc = sum(BACC) / val_nums
                bacc_std = np.array(BACC).std()

                # mean_dice_jacc = mean_dice * 0.5 + mean_jacc * 0.5
                print('VAL ______________________')
                print('Epoch : %d, mean_dice : %.4f, std : %.4f, mean_jacc : %.4f, std : %.4f, mean_bacc : %.4f ' %
                      (epoch, mean_dice, dice_std, mean_jacc, jacc_std, mean_bacc))

            early_stop(mean_dice, net, epoch + 1)

            if early_stop.early_stop:
                print('early stop')
                break
    print('----------------------------------------------------------')
    print('save epoch {}'.format(early_stop.save_epoch))
    print('stoped epoch {}'.format(epoch))
    print('trained task {}'.format(task))
    print('trained number id {}'.format(number))
    print('----------------------------------------------------------')

def test_main():
    pth = pth_save_name
    model = torch.load(pth).cuda()

    Loader, nums = getBiDataLoder(path=path, isTrain='test', task=task, is_shuffle=False, sampling_rate=sampling_rate,
                                  crop_size=crop_size, is_norm=is_norm, use_pro=False)
    test(Loader, nums, model)
    countparam(model)

def test(Loader, nums, net):
    overlip = crop_size - 100
    Dice = []
    JACC = []
    BACC = []
    flag = 1
    with torch.no_grad():
        for _, (data, octa_slice, gt, gt_slices, pro_data) in enumerate(Loader):
            data = data.cuda()
            octa_slice = octa_slice.cuda()
            gt = gt.cuda()
            gt_slices = gt_slices.cuda()
            pro_data = pro_data.cuda()
            pred, _, _ = net(data, octa_slice, pro_data)
            # 训练出来的是一个416 * 416的图，我们支取中间的400*400
            # pred = TTF.center_crop(pred, (400, 400))
            pred = torch.sigmoid(pred)
            pred[pred > 0.5] = 1
            pred[pred < 0.5] = 0
            pad_size = 200 - crop_size
            if task == 'FAZ':
                gt = torch.nn.functional.pad(gt, pad=(pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
                pred = torch.nn.functional.pad(pred, pad=(pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            dice = diceCoeffv2(pred[:, 1:2, :], gt[:, 0:1, :], activation=None).cpu()
            Dice.append(dice)
            # jac
            jacc = jaccardv2(pred[:, 1:2, :], gt[:, 0:1, :]).cpu()
            JACC.append(jacc)

            bacc = balance_acc(pred[:, 1:2, :], gt[:, 0:1, :]).cpu()
            BACC.append(bacc)

            pred_img = pred[0][1]
            pred_img = pred_img.detach().cpu().numpy()
            pred_img = pred_img * 255
            # save_img(save_img_path, pred_img, 200 + flag, dice)
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
    path = os.path.join(pth, str(number)+'_'+str(dice) + '.jpeg')
    im.convert('1').save(path)
if __name__ == '__main__':
    main()
    print('now number : ', number)
    print('fusion_net : ', fusion_net, 'task : ', task, 'plm_nums', plm_nums, 'now aux rate', aux_rate)

