import random
import time

import cv2
import numpy as np
import torch
from albumentations import DualTransform, to_tuple, ImageOnlyTransform, RandomSizedCrop
from albumentations.augmentations.functional import preserve_channel_dim, _maybe_process_in_chunks

_DEFAULT_ALPHASTD = 0.1
_DEFAULT_EIGVAL = np.array([0.2175, 0.0188, 0.0045])
_DEFAULT_EIGVEC = np.array([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]])
_DEFAULT_BCS = [0.2, 0.2, 0.2]


class SafeRotate(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
            self,
            limit=90,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            value=None,
            mask_value=None,
            always_apply=False,
            p=0.5,
    ):
        super(SafeRotate, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return rotate_im(img, angle, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, angle=0, **params):
        return rotate_im(img, angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def get_params(self):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}

    def apply_to_bbox(self, bbox, angle=0, **params):
        raise NotImplementedError()

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        raise NotImplementedError()

    def get_transform_init_args_names(self):
        return ("limit", "interpolation", "border_mode", "value", "mask_value")


@preserve_channel_dim
def rotate_im(img, angle, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=M, dsize=(nW, nH), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    return warp_fn(img)


class Lighting(ImageOnlyTransform):
    """Random Lighting

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, alphastd=_DEFAULT_ALPHASTD, eigval=_DEFAULT_EIGVAL, eigvec=_DEFAULT_EIGVEC, always_apply=False,
                 p=0.5):
        super(Lighting, self).__init__(always_apply, p)
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def apply(self, image, alpha=np.array([0, 0, 0]), **params):
        if self.alphastd == 0.:
            return image

        rgb = (self.eigvec * alpha * self.eigval).sum(axis=1)
        return np.clip((255 * (image / 255. + rgb.reshape(1, 1, 3))).astype(np.int32), 0, 255).astype(np.uint8)

    def get_params(self):
        return {
            "alpha": np.random.normal(torch.zeros(3), self.alphastd)
        }

    def get_transform_init_args_names(self):
        return ("alphastd", "eigval", "eigvec")


class RandomSizedCropAroundBbox(RandomSizedCrop):
    @property
    def targets_as_params(self):
        return ['rectangles', 'image']

    def get_params_dependent_on_targets(self, params):
        rectangles = params['rectangles']
        img_height, img_width = params['image'].shape[:2]
        rm = random.Random()
        rm.seed(time.time_ns())
        crop_height = rm.randint(self.min_max_height[0], self.min_max_height[1])
        crop_width = int(crop_height * self.w2h_ratio)

        if rectangles:
            x, y, w, h = rm.choice(rectangles)
            min_x_start = max(x + (w / 2 if w >= crop_width else w) - crop_width, 0)
            min_y_start = max(y + (h / 2 if h >= crop_height else h) - crop_height, 0)
            max_x_start = min(x + (w / 2 if w >= crop_width else 0), img_width - crop_width)
            max_y_start = min(y + (h / 2 if h >= crop_height else 0), img_height - crop_height)
            if max_x_start < min_x_start:
                min_x_start, max_x_start = max_x_start, min_x_start
            if max_y_start < min_y_start:
                min_y_start, max_y_start = max_y_start, min_y_start
            start_y = rm.randint(int(min_y_start), int(max_y_start)) / img_height
            start_x = rm.randint(int(min_x_start), int(max_x_start)) / img_width
        else:
            start_y = rm.random()
            start_x = rm.random()
        return {'h_start': (start_y * img_height) / (img_height - crop_height),
                'w_start': (start_x * img_width) / (img_width - crop_width),
                'crop_height': crop_height,
                'crop_width': int(crop_height * self.w2h_ratio)}
