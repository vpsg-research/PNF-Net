import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from net.PNFNet import Net
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()



def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, edges) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            edges = Variable(edges).cuda()

            preds = model(images)
            loss_init = structure_loss(preds[1], gts) + structure_loss(preds[2], gts) + structure_loss(preds[3], gts)

            loss_final = structure_loss(preds[0], gts)

            loss = loss_final + loss_init

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss_init: {:.4f} Loss_final: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, 0, loss_final.data))

                with open(opt.save_log, 'a') as f:
                    f.write('\n')
                    f.write('[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss_init: {:.4f} '
                    'Loss_final: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, 0, loss_final.data))
                    f.write('\n')
               
        loss_all /= epoch_step

        with open(opt.save_log, 'a') as f:
            f.write('\n')
            f.write('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
            f.write('\n')
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if  epoch % 10 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=101, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=416, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0,1', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='/home/lsl/lsl-IML/IML-DS/train/',
                        help='the training rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='/home/lsl/lsl-IML/TCSVT2026/weight/',
                        help='the path to save model and log')
    opt = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    num_gpus = len(opt.gpu_id.split(','))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    model = Net().to(device)

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Tp/',
                              gt_root=opt.train_root + 'Gt/',
                              edge_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4)
    val_loader = test_dataset(image_root=opt.val_root + 'Tp/',
                              gt_root=opt.val_root + 'Gt/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)


    opt.save_log = save_path + 'log.log'
    with open(opt.save_log, 'a') as f:
        f.write('-----------')
        f.write('\n')
        f.write('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))
        f.write('\n')

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 0
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer)