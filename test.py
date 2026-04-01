import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from net.PNFNet import Net
from utils.data_val import get_loader, test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='training dataset size')
parser.add_argument('--gpu_id', type=str, default='0,1', help='train use gpu')
parser.add_argument('--pth_path', type=str, default='......')
opt = parser.parse_args()

for _data_name in ['Columbia', 'C1','Coverage','IMD2020','NC16','Korus']:
    data_path = '/home/lsl/lsl-IML/IML-DS/test/{}'.format(_data_name)
    save_path = '/home/lsl/lsl-IML/TCSVT2026/result/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = Net()

    
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    
    os.makedirs(save_path, exist_ok=True)
    #os.makedirs(save_path+'edge/', exist_ok=True)
    image_root = '{}/Tp/'.format(data_path)
    gt_root = '{}/Gt/'.format(data_path)
    
    # test_loader = test_dataset(image_root, gt_root, opt.testsize)

    test_loader = test_dataset(image_root=image_root,
                              gt_root=gt_root,
                              testsize=opt.testsize)
    for i in range(test_loader.size):
        image, gt, name, image_for_post = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        with torch.no_grad():
            res= model(image)
        res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        imageio.imwrite(save_path+name, (res*255).astype(np.uint8))