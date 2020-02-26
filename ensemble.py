import argparse
import os
from os import cpu_count

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import skimage
import skimage.io


from multiprocessing.pool import Pool

import numpy as np
import cv2
cv2.setNumThreads(0)


def average_strategy(images):
    return np.average(images, axis=0)


def hard_voting(images):
    rounded = np.round(images / 255.)
    return np.round(np.sum(rounded, axis=0) / images.shape[0]) * 255.


def ensemble_image(params):
    file, dirs, ensembling_dir, strategy = params
    images = []
    for dir in dirs:
        file_path = os.path.join(dir, file)
        try:
            images.append(skimage.io.imread(file_path))
        except:
            images.append(skimage.io.imread(file_path))
    images = np.array(images)

    if strategy == 'average':
        ensembled = average_strategy(images)
    elif strategy == 'hard_voting':
        ensembled = hard_voting(images)
    else:
        raise ValueError('Unknown ensembling strategy')
    skimage.io.imsave(os.path.join(ensembling_dir, file), ensembled.astype(np.uint8))


def ensemble(dirs, strategy, ensembling_dir, n_threads, mode="localization"):
    files = os.listdir(dirs[0])
    params = []

    for file in files:
        if file.endswith(mode + ".png"):
            params.append((file, dirs, ensembling_dir, strategy))
    pool = Pool(n_threads)
    pool.map(ensemble_image, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Ensemble masks")
    arg = parser.add_argument
    arg('--ensembling_cpu_threads', type=int, default=cpu_count())
    arg('--ensembling_dir', type=str,)
    arg('--strategy', type=str, default='average')
    arg('--folds_dir', type=str, )
    arg('--dirs_to_ensemble', nargs='+')
    arg('--mode',  default='localization')
    args = parser.parse_args()

    folds_dir = args.folds_dir
    dirs = [os.path.join(folds_dir, d) for d in args.dirs_to_ensemble]
    for d in dirs:
        if  not os.path.exists(d):
            raise ValueError(d + " doesn't exist")
    os.makedirs(args.ensembling_dir, exist_ok=True)
    ensemble(dirs, args.strategy, args.ensembling_dir, args.ensembling_cpu_threads, mode=args.mode)
