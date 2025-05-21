import random

import numpy as np
import torch
import torch.nn.functional as F
from nycu_cv_hw4.constants import DEBUG_DIR_PATH
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


# crop an image to the multiple of base
def crop_img(image, base=64):
    h, w = image.shape[:2]
    crop_h = h % base
    crop_w = w % base
    return image[
        crop_h // 2 : h - crop_h + crop_h // 2,
        crop_w // 2 : w - crop_w + crop_w // 2,
        :,
    ]


def compute_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


def random_crop_patch(img_1, img_2, patch_size: int):
    H = img_1.shape[0]
    W = img_1.shape[1]
    ind_H = random.randint(0, H - patch_size)
    ind_W = random.randint(0, W - patch_size)

    patch_1 = img_1[
        ind_H : ind_H + patch_size,
        ind_W : ind_W + patch_size,
    ]
    patch_2 = img_2[
        ind_H : ind_H + patch_size,
        ind_W : ind_W + patch_size,
    ]

    return patch_1, patch_2


def set_seed(seed: int = 42):
    random.seed(seed)  # Python random 模組
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # 單 GPU
    torch.cuda.manual_seed_all(seed)  # 多 GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_pair(image_id, de_patch, clean_patch):
    grid = make_grid(torch.stack([de_patch, clean_patch]), nrow=2)  # [3, H, W]
    pil_img = to_pil_image(grid)
    pil_img.save(DEBUG_DIR_PATH / f"{image_id}.png")
