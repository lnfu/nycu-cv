# TODO crop_img (utils.py) 這次作業都是 256x256 所以不用處理
from pathlib import Path

import kornia.augmentation as K
import numpy as np
import torch
from kornia.augmentation import AugmentationSequential
from kornia.utils import image_to_tensor
from nycu_cv_hw4.utils import random_crop_patch
from PIL import Image
from torch.utils.data import Dataset


class PromptTrainDataset(Dataset):
    train: bool
    patch_size: int

    def __init__(
        self,
        data_dir: Path,
        de_types=["derain", "desnow"],
        train: bool = True,
        patch_size: int = 128,
    ):
        super(PromptTrainDataset, self).__init__()
        self.train = train
        self.patch_size = patch_size

        if train:
            self.consistent_augment = AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                same_on_batch=True,
            )
        else:
            self.consistent_augment = None

        self.de_dict = {"derain": 0, "desnow": 1}

        # self.de_types = self.de_dict.keys()  # TODO remove
        self.de_types = de_types
        total_de_paths = list()
        total_clean_paths = list()
        total_de_types = list()
        total_image_ids = list()

        for de_type in self.de_types:
            de_type = de_type[2:]  # derain -> rain

            de_paths = data_dir.glob(f"degraded/{de_type}-*.png")
            # clean_paths = DATA_DIR.glob(f"clean/{de_type}_clean-*.png")

            total_de_paths.extend(list(de_paths))
            # total_clean_paths.extend(list(clean_paths))

        for de_path in total_de_paths:
            de_type = de_path.stem.split("-")[0]
            image_id = de_path.stem.split("-")[-1]
            clean_path = data_dir / "clean" / f"{de_type}_clean-{image_id}.png"

            total_de_types.append(f"de{de_type}")
            total_image_ids.append(image_id)
            total_clean_paths.append(clean_path)

        self.samples = list(
            zip(
                total_image_ids,
                total_de_types,
                total_de_paths,
                total_clean_paths,
            )
        )

        # pprint(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id, de_type, de_path, clean_path = sample

        de_img = np.array(Image.open(de_path).convert("RGB"))  # 0 - 255
        clean_img = np.array(Image.open(clean_path).convert("RGB"))  # 0 - 255

        # H, W, C
        if self.train:
            de_patch, clean_patch = random_crop_patch(
                de_img, clean_img, self.patch_size
            )
        else:
            de_patch, clean_patch = de_img, clean_img

        # C, H, W
        de_patch = image_to_tensor(de_patch, keepdim=False).float() / 255.0
        clean_patch = (
            image_to_tensor(clean_patch, keepdim=False).float() / 255.0
        )

        if self.consistent_augment:
            stacked = torch.cat([de_patch, clean_patch], dim=0)  # [2, 3, H, W]
            stacked = self.consistent_augment(stacked)
            de_patch, clean_patch = stacked[0], stacked[1]
        else:
            de_patch, clean_patch = de_patch[0], clean_patch[0]

        # visualize_pair(image_id, de_patch, clean_patch)

        return [image_id, self.de_dict[de_type]], de_patch, clean_patch

    def __len__(self):
        return len(self.samples)
