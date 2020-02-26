import argparse
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from collections import namedtuple

import cv2
import numpy as np
import skimage.io
import torch
from albumentations.pytorch.transforms import img_to_tensor
from skimage import measure
from skimage.morphology import watershed

import models
from tools.config import load_config

ModelConfig = namedtuple("ModelConfig", "config_path weight_path type weight")
weight_path = "weights"
configs = [
    ModelConfig("configs/d92_loc.json", "localization_dpn_unet_dpn92_0_best_dice", "localization", 1),
    ModelConfig("configs/d92_loc.json", "localization_dpn_unet_dpn92_1_best_dice", "localization", 1),
    ModelConfig("configs/d92_loc.json", "localization_dpn_unet_dpn92_2_best_dice", "localization", 1),
    ModelConfig("configs/d92_loc.json", "localization_dpn_unet_dpn92_3_best_dice", "localization", 1),
     ModelConfig("configs/d161_loc.json", "localization_densenet_unet_densenet161_3_0_best_dice", "localization", 1),
    ModelConfig("configs/d161_loc.json", "localization_densenet_unet_densenet161_3_1_best_dice", "localization", 1),

    ModelConfig("configs/d92_softmax.json", "softmax_dpn_seamese_unet_shared_dpn92_0_best_xview", "damage", 1),
    ModelConfig("configs/d92_softmax.json", "softmax_dpn_seamese_unet_shared_dpn92_2_best_xview", "damage", 1),
    ModelConfig("configs/d92_softmax.json", "pseudo_dpn_seamese_unet_shared_dpn92_0_best_xview", "damage", 1),
    ModelConfig("configs/d92_softmax.json", "pseudo_dpn_seamese_unet_shared_dpn92_2_best_xview", "damage", 1),

    ModelConfig("configs/d161_softmax.json", "softmax_densenet_seamese_unet_shared_densenet161_0_best_xview",
                "damage", 1),
    ModelConfig("configs/d161_softmax.json", "softmax_densenet_seamese_unet_shared_densenet161_2_best_xview",
                "damage", 1),
    ModelConfig("configs/d161_softmax.json", "pseudo_densenet_seamese_unet_shared_densenet161_0_best_xview",
                "damage", 1),
    ModelConfig("configs/d161_softmax.json", "pseudo_densenet_seamese_unet_shared_densenet161_2_best_xview",
                "damage", 1),

    ModelConfig("configs/se50_softmax.json", "pseudo_scseresnext_seamese_unet_shared_seresnext50_0_best_xview",
                "damage", 1),
    ModelConfig("configs/se50_softmax.json", "pseudo_scseresnext_seamese_unet_shared_seresnext50_1_best_xview",
                "damage", 1),
    ModelConfig("configs/se50_softmax.json", "pseudo_scseresnext_seamese_unet_shared_seresnext50_2_best_xview",
                "damage", 1),
    ModelConfig("configs/se50_softmax.json", "pseudo_scseresnext_seamese_unet_shared_seresnext50_3_best_xview",
                "damage", 1),

    ModelConfig("configs/b2_softmax.json",
                "softmax_sampling_efficient_seamese_unet_shared_efficientnet-b2_0_best_xview", "damage", 1),
    ModelConfig("configs/b2_softmax.json",
                "softmax_sampling_efficient_seamese_unet_shared_efficientnet-b2_1_best_xview", "damage", 1),

    ModelConfig("configs/r101_softmax_sgd.json", "sgd_resnext_seamese_unet_shared_resnext101_0_best_xview", "damage", 2)
]


def predict_localization(image, config: ModelConfig):
    conf = load_config(config.config_path)
    model = models.__dict__[conf['network']](seg_classes=1, backbone_arch=conf['encoder'])
    checkpoint_path = os.path.join(weight_path, config.weight_path)
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()})

    model.eval()

    model = model.cpu()
    print("predicting", config)
    with torch.no_grad():
        image = img_to_tensor(image, conf["input"]["normalize"]).cpu().numpy()
        if "dpn" in config.weight_path:
            image = np.pad(image, [(0, 0), (16, 16), (16, 16)], mode='reflect')

        images = np.array(
            [
                image,
                image[:, ::-1, :],
                image[:, :, ::-1],
                image[:, ::-1, ::-1],
            ])
        images = torch.from_numpy(images).cpu().float()
        prediction_masks = []
        for i in range(4):

            logits = model(images[i:i + 1])
            preds = torch.sigmoid(logits).cpu().numpy()

            pred = preds[0]
            if i == 1:
                pred = preds[0].copy()[:, ::-1, :]
            if i == 2:
                pred = preds[0].copy()[:, :, ::-1]
            if i == 3:
                pred = preds[0].copy()[:, ::-1, ::-1]
            prediction_masks.append(pred)

        preds = np.average(prediction_masks, axis=0)
        if "dpn" in config.weight_path:
            preds = preds[:, 16:-16, 16:-16]
        return preds


def predict_localization_ensemble(pre_path):
    image = cv2.imread(pre_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    preds = []
    for model_config in configs:
        if model_config.type == "localization":
            preds.append((predict_localization(image, model_config) * 255).astype(np.uint8))
    return np.average(preds, axis=0)


def predict_damage_ensemble(pre_path, post_path):
    image_pre = cv2.imread(pre_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    image_post = cv2.imread(post_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    image = np.concatenate([image_pre, image_post], axis=-1)
    preds = []
    for model_config in configs:
        if model_config.type == "damage":
            damage = (predict_damage(image, model_config) * 255).astype(np.uint8)
            preds.append(damage)
            if model_config.weight == 2:
                preds.append(damage)
    return np.average(preds, axis=0)


def predict_damage(image, config: ModelConfig):
    conf = load_config(config.config_path)
    model = models.__dict__[conf['network']](seg_classes=5, backbone_arch=conf['encoder'])
    checkpoint_path = os.path.join(weight_path, config.weight_path)
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict({k[7:]: v for k, v in checkpoint['state_dict'].items()})

    model.eval()

    model = model.cpu()
    print("predicting", config)
    with torch.no_grad():
        image = img_to_tensor(image, conf["input"]["normalize"]).cpu().numpy()
        images = np.array(
            [
                image,
                image[:, ::-1, :],
                image[:, :, ::-1],
                image[:, ::-1, ::-1],
            ])
        images = torch.from_numpy(images).cpu().float()
        prediction_masks = []
        for i in range(4):

            logits = model(images[i:i + 1])
            preds = torch.softmax(logits, dim=1).cpu().numpy()

            pred = preds[0]
            if i == 1:
                pred = preds[0].copy()[:, ::-1, :]
            if i == 2:
                pred = preds[0].copy()[:, :, ::-1]
            if i == 3:
                pred = preds[0].copy()[:, ::-1, ::-1]
            prediction_masks.append(pred)

        preds = np.average(prediction_masks, axis=0)
        return preds


def label_mask(loc, labels, intensity, mask, seed_threshold=0.8):
    av_pred = 1 * (loc > seed_threshold)
    av_pred = av_pred.astype(np.uint8)

    y_pred = measure.label(av_pred, neighbors=8, background=0)

    nucl_msk = (1 - loc)
    nucl_msk = nucl_msk.astype('uint8')
    y_pred = watershed(nucl_msk, y_pred, mask=mask, watershed_line=False)
    props = measure.regionprops(y_pred)
    for i in range(1, np.max(y_pred)):
        reg_labels = labels[y_pred == i]
        unique, counts = np.unique(reg_labels, return_counts=True)
        max_idx = np.argmax(counts)
        out_label = unique[max_idx]
        if out_label > 0:
            prop = props[i - 1]
            if counts[max_idx] > 0.6 * sum(counts) \
                    and prop.eccentricity < 1.5 \
                    and prop.euler_number == 1:
                labels[(y_pred == i) & (intensity < 0.6)] = out_label

    return y_pred


def post_process(loc, damage, out_loc, out_damage):
    localization = loc/255.
    damage = damage/255.
    #damage = np.moveaxis(damage[1:, ...], 0, -1)/255.
    first = np.zeros((1024, 1024, 1))
    first[:, :, 0] = 1 - np.sum(damage, axis=2)
    first[:, :, :] *= 0.8

    damage_pred = np.concatenate([first, damage], axis=-1)

    damage_pred[:, :, 2] *= 2.
    damage_pred[:, :, 3] *= 2.
    damage_pred[:, :, 4] *= 2.

    argmax = np.argmax(damage_pred, axis=-1)
    loc = 1 * ((localization > 0.25) | (argmax > 0))
    argmax = np.argmax(damage, axis=-1) + 1
    max = np.max(damage, axis=-1)
    label_mask(localization, argmax, max, loc)
    cv2.imwrite(out_loc, loc)
    cv2.imwrite(out_damage, argmax)


def main():
    parser = argparse.ArgumentParser("Xview Predictor")
    arg = parser.add_argument
    arg('--pre', type=str, help='Path to pre test image')
    arg('--post', type=str, help='Path to post test image')
    arg('--out-loc', type=str, help='Path to output localization image')
    arg('--out-damage', type=str, help='Path to output damage image')
    args = parser.parse_args()
    localization = predict_localization_ensemble(args.pre)
    skimage.io.imsave("_localization.png", localization[0, :, :].astype(np.uint8))
    damage = predict_damage_ensemble(args.pre, args.post)
    preds = (np.moveaxis(damage, 0, -1)).astype(np.uint8)
    cv2.imwrite("_damage.png", cv2.cvtColor(preds[:, :, 1:], cv2.COLOR_RGBA2BGRA))
    damage = skimage.io.imread("_damage.png")
    localization = cv2.imread( "_localization.png", cv2.IMREAD_GRAYSCALE)
    post_process(localization, damage, args.out_loc,  args.out_damage)



if __name__ == '__main__':
    main()
