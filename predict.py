import argparse
import os
import warnings

import numpy as np
import skimage
import skimage.io
import torch
from albumentations import Compose, PadIfNeeded
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from dataset.xview_dataset import XviewSingleDataset
from tools.config import load_config

warnings.simplefilter("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Xview Predictor")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', default='configs/r50.json', help='path to configuration file')
    arg('--data-path', type=str, default='/home/selim/datasets/xview/train/', help='Path to test images')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--dir', type=str, default='../predictions/xview/r50_dice')
    arg('--mask_dir', type=str, default='../predictions/masks')
    arg('--model', type=str, default='weights2/spacenet_resnext_unet_resnext50_0_best_dice')

    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    conf = load_config(args.config)
    model = models.__dict__[conf['network']](seg_classes=5, backbone_arch=conf['encoder'])
    model = torch.nn.DataParallel(model).cuda()
    print("=> loading checkpoint '{}'".format(args.model))
    checkpoint = torch.load(args.model, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    transforms = Compose([
    ])
    dataset = XviewSingleDataset(data_path=args.data_path, transforms=transforms, mode="val", normalize=conf['input'].get('normalize', None))
    data_loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=False)
    with torch.no_grad():
        for sample in tqdm(data_loader):
            ids = sample['img_name']
            image = sample['image'].numpy()[0]
            images = np.array(
                [
                    image,
                    image[:, ::-1, :],
                    image[:, :, ::-1],
                    image[:, ::-1, ::-1],
                ])
            images = torch.from_numpy(images).cuda().float()
            logits = model(images)
            preds = torch.sigmoid(logits).cpu().numpy()

            prediction_masks = []
            for i in range(4):
                pred = preds[i]
                if i == 1:
                    pred = preds[i].copy()[:, ::-1, :]
                if i == 2:
                    pred = preds[i].copy()[:, :, ::-1]
                if i == 3:
                    pred = preds[i].copy()[:, ::-1, ::-1]
                prediction_masks.append(pred)
            preds = np.average(prediction_masks, axis=0)

            preds = (np.moveaxis(preds, 0, -1) * 255).astype(np.uint8)
            skimage.io.imsave(args.dir + "/" + ids[0] + "_localization.png", preds[:, :, 4])
            skimage.io.imsave(args.dir + "/" + ids[0] + "_damage.png", preds[:, :, :-1])
