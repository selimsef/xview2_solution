import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.functional import img_to_tensor
from skimage.measure import label
from torch.utils.data import Dataset


class XviewSingleDataset(Dataset):
    def __init__(self, data_path, mode, fold=0, folds_csv='folds.csv', equibatch=False, transforms=None, normalize=None,
                 multiplier=1):
        super().__init__()
        self.data_path = data_path
        self.mode = mode

        self.names = sorted(os.listdir(os.path.join(self.data_path, "images")))
        df = pd.read_csv(folds_csv, dtype={'id': object})
        self.df = df
        self.normalize = normalize
        self.fold = fold
        self.equibatch = equibatch
        if self.mode == "train":
            ids = df[df['fold'] != fold]['id'].tolist()
            nondamage = df[(df['fold'] != fold) & (df['nondamage'] == True)]['id'].tolist()
            minor = df[(df['fold'] != fold) & (df['minor'] == True)]['id'].tolist()
            major = df[(df['fold'] != fold) & (df['major'] == True)]['id'].tolist()
            destroyed = df[(df['fold'] != fold) & (df['destroyed'] == True)]['id'].tolist()
            empty = df[(df['fold'] != fold) & (df['empty'] == True)]['id'].tolist()

            self.group_names = {
                "nondamage1": nondamage,
                "nondamage": nondamage,
                "minor": minor,
                "major": major,
                "destroyed": destroyed,
                "empty": empty,
            }
            self.group_ids = list(self.group_names.keys())
            if not self.equibatch:
                ids.extend(minor)
                ids.extend(major)
                ids.extend(destroyed)
        else:
            ids = list(set(df[df['fold'] == fold]['id'].tolist()))
        self.transforms = transforms
        self.names = ids

        if mode == "train":
            self.names = self.names * multiplier
        self.cache = {}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        if self.mode == 'train' and self.equibatch:
            group_id = self.group_ids[idx % len(self.group_ids)]
            name = random.choice(self.group_names[group_id])
        else:
            group_id = "unknown"
            name = self.names[idx]
        pre_img_path = os.path.join(self.data_path, "images", name + "_pre_disaster.png")
        post_img_path = os.path.join(self.data_path, "images", name + "_post_disaster.png")
        image_pre = cv2.imread(pre_img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        image_post = cv2.imread(post_img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        mask_pre = cv2.imread(os.path.join(self.data_path, "masks", name + "_pre_disaster.png"), cv2.IMREAD_GRAYSCALE)
        mask_post = cv2.imread(os.path.join(self.data_path, "masks", name + "_post_disaster.png"), cv2.IMREAD_GRAYSCALE)

        rectangles = self.cache.get(self.names[idx], [])
        if not rectangles:
            self.add_boxes(label(mask_post == 2).astype(np.uint8), rectangles)
        if rectangles:
            self.cache[self.names[idx]] = rectangles

        mask = np.stack([mask_pre, mask_post, mask_post], axis=-1)
        sample = self.transforms(image=image_pre, image1=image_post, mask=mask, img_name=name, rectangles=rectangles)
        image = np.concatenate([sample['image'], sample['image1']], axis=-1)
        sample['img_name'] = name
        sample['group_id'] = group_id
        mask = np.zeros((5, *sample["mask"].shape[:2]))
        for i in range(1, 5):
            mask[i - 1, sample["mask"][:, :, 1] == i] = 1
        mask[4] = sample["mask"][:, :, 0] / 255
        del sample["image1"]
        sample['original_mask'] = torch.from_numpy(np.ascontiguousarray(sample["mask"][:, :, 1]))
        sample['mask'] = torch.from_numpy(np.ascontiguousarray(mask)).float()
        sample['image'] = img_to_tensor(np.ascontiguousarray(image), self.normalize)
        return sample

    def add_boxes(self, labels, rectangles):
        max_label = np.max(labels)
        for i in range(1, max_label + 1):
            obj_mask = np.zeros_like(labels)
            obj_mask[labels == i] = 255

            contours, hierarchy = cv2.findContours(obj_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                points = cv2.boundingRect(cnt)
                rectangles.append(points)



class XviewSingleDatasetTest(Dataset):
    def __init__(self, data_path, transforms=None, normalize=None):
        super().__init__()
        self.data_path = data_path
        self.names = list(set([os.path.splitext(f)[0].replace("test_post_", "").replace("test_pre_", "") for f in
                               sorted(os.listdir(os.path.join(self.data_path, "images")))]))
        self.normalize = normalize
        self.transforms = transforms

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        pre_img_path = os.path.join(self.data_path, "images", "test_pre_" + name + ".png")
        post_img_path = os.path.join(self.data_path, "images", "test_post_" + name + ".png")

        image_pre = cv2.imread(pre_img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        image_post = cv2.imread(post_img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        image = np.concatenate([image_pre, image_post], axis=-1)
        sample = {}
        sample['img_name'] = name
        sample['image'] = img_to_tensor(np.ascontiguousarray(image), self.normalize)
        return sample
