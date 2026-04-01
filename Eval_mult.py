import os
import cv2
import numpy as np
import torch
import warnings
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        iou = 1
    return f1, iou


# === 路径匹配函数 ===
def get_gt_path(file, path_gt):
    if "NC16" in path_gt:
        return path_gt + file
    elif "C1" in path_gt:
        return path_gt + file[:-4] + '_gt.png'
    elif "Coverage" in path_gt:
        return path_gt + file[:-5] + 'forged.tif'
    elif "Columbia" in path_gt:
        return path_gt + file[:-4] + '_edgemask.png'
    elif "2020" in path_gt:
        return path_gt + file[:-4] + '_mask.png'
    elif "Korus" in path_gt:
        return path_gt + file[:-4] + '.PNG'
    elif "Coco" in path_gt or "In-the-Wild" in path_gt:
        return path_gt + file[:-4] + '.png'
    elif "DSO" in path_gt:
        return path_gt + file[:-4] + '_gt.png'
    else:
        return path_gt + file[:-5] + '_gt.png'


# === 单张图处理函数 ===
def process_single_image(args):
    file, path_pre, path_gt = args
    try:
        pre = cv2.imread(path_pre + file)
        gt_path = get_gt_path(file, path_gt)
        gt = cv2.imread(gt_path)

        if pre is None or gt is None:
            return None

        H, W, C = pre.shape
        Hg, Wg, C = gt.shape
        if H != Hg or W != Wg:
            gt = cv2.resize(gt, (W, H))
            gt[gt > 127] = 255
            gt[gt <= 127] = 0

        # AUC
        auc = None
        if np.max(gt) != np.min(gt):
            auc = roc_auc_score((gt.reshape(H * W * C) / 255).astype('int'),
                                pre.reshape(H * W * C) / 255.)

        pre[pre > 127] = 255
        pre[pre <= 127] = 0
        f1, iou = metric(pre / 255, gt / 255)
        return (auc, f1, iou)
    except Exception as e:
        print(f"[Error] {file}: {e}")
        return None


# === 并行评估主函数 ===
def evaluate(path_pre, path_gt, dataset_name, record_txt):
    flist = sorted(os.listdir(path_pre))
    args_list = [(f, path_pre, path_gt) for f in flist]

    print(f"🚀 使用 {min(16, cpu_count())} 核并行处理 {len(flist)} 张图像")
    with Pool(processes=min(16, cpu_count())) as pool:
        results = list(tqdm(pool.imap(process_single_image, args_list), total=len(args_list)))

    aucs, f1s, ious = [], [], []
    num_valid = 0
    for r in results:
        if r is not None:
            auc, f1, iou = r
            if auc is not None:
                aucs.append(auc)
            f1s.append(f1)
            ious.append(iou)
            num_valid += 1

    print(f"✅ {dataset_name}")
    print('Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f' % (
        np.mean(aucs), np.mean(f1s), np.mean(ious)))
    print(f'🧮 评估了 {num_valid} 张图')

    with open(record_txt, "a") as f:
        f.write(dataset_name + "\n")
        f.write('Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f\n' %
                (np.mean(aucs), np.mean(f1s), np.mean(ious)))
    return np.mean(aucs), np.mean(f1s), np.mean(ious)


# === 入口 ===
if __name__ == "__main__":
    save_path = '/home/lsl/lsl-IML/TCSVT2026/result/epoch_40_Online/CASIA_Whatsapp/'
    path_gt = '/home/lsl/lsl-IML/IML-DS/Online_test/CASIA_Facebook/Gt/'
    record_txt = "/home/lsl/lsl-IML/TCSVT2026/result/epoch_40/eval.txt"

    with open(record_txt, "a") as f:
        f.write("\n")

    name = path_gt.split("/")[-3]
    if "sam" in save_path:
        name = name + " sam"

    evaluate(save_path, path_gt, name, record_txt)
