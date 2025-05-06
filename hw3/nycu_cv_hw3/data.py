import json
from pathlib import Path
from typing import Tuple

import albumentations as A
import numpy as np
import tifffile
import torch
from albumentations.pytorch import ToTensorV2
from nycu_cv_hw3.config import settings
from nycu_cv_hw3.constants import DATA_DIR_PATH
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes

transform = A.Compose(
    [
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(transpose_mask=True),
    ],
    additional_targets={"masks": "masks"},
)

augmentated_transform = A.Compose(
    [
        # geometry
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.0, 0.1),
            rotate=(-15, 15),
            p=0.5,
        ),
        # color
        A.RGBShift(
            r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.5
        ),
        # noise
        A.GaussNoise(),
        # default
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(transpose_mask=True),
    ],
    additional_targets={"masks": "masks"},
)


class TrainDataset(Dataset):
    """
    {
        "boxes",
        "labels",
        "masks",
        "image_id",
        "area",
        "iscrowd"
    }
    """

    def __init__(self, data_dir: Path, transform):
        self.transform = transform

        # each sample is a directory
        self.sample_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
        self.sample_dirs.sort()

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        sample_dir = self.sample_dirs[idx]

        # H x W x 4
        image = tifffile.imread(sample_dir / "image.tif").astype(np.uint8)
        image = image[:, :, :3]  # RGBA -> RGB

        labels = []
        masks = []

        for path in sample_dir.glob("class*.tif"):
            # e.g. class3.tif -> 3
            label = int(path.stem.replace("class", ""))
            # 不能用 uint8 讀, 不然會最大 255
            mask = tifffile.imread(path)  # H x W
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]
            for instance_id in instance_ids:
                labels.append(label)
                # uint8 是因為 transform, 實際上是 bool
                masks.append((mask == instance_id).astype(np.uint8))

        if self.transform:
            transformed = self.transform(image=image, masks=masks)
            image = transformed["image"]
            masks = transformed["masks"]
        else:
            # TODO
            print("ERROR")
            exit(0)

        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)

        # 進行 data augmentation 之後有可能導致 mask 變成全 0, 需要 filter
        valid = masks.flatten(1).sum(1) > 0
        masks = masks[valid]
        labels = labels[valid]

        boxes = masks_to_boxes(masks)

        # 移除 box w/h 是 0 的
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths > 0) & (heights > 0)
        boxes = boxes[valid]
        masks = masks[valid]
        labels = labels[valid]

        image_id = torch.tensor([idx])
        # area = masks.sum(dim=[1, 2]) if len(masks) > 0 else torch.tensor([])
        # iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        return image, {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            # "area": area,
            # "iscrowd": iscrowd,
            "image_id": image_id,
            # "image_name": sample_dir.stem,
        }


class InferenceDataset(Dataset):

    def __init__(self, data_dir: Path, metadata_path: Path, transform):
        self.transform = transform
        self.image_paths = sorted(data_dir.glob("*.tif"))
        assert len(self.image_paths) > 0, f"No .tif files found in {data_dir}"

        with open(metadata_path, "r") as f:
            raw_metadata = json.load(f)
        self.metadata = {
            Path(entry["file_name"]).stem: {
                "id": entry["id"],
                "width": entry["width"],
                "height": entry["height"],
            }
            for entry in raw_metadata
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        image_name = path.stem  # no ".tif"

        image = tifffile.imread(path).astype(np.uint8)  # shape: [H, W, 4]
        image = image[:, :, :3]  # RGBA -> RGB

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # TODO
            print("ERROR")
            exit(0)

        return (
            image,
            image_name,
            self.metadata[image_name]["id"],
            self.metadata[image_name]["width"],
            self.metadata[image_name]["height"],
        )


all_dataset = TrainDataset(
    DATA_DIR_PATH / "train", transform=augmentated_transform
)

# TODO config
train_size = int(0.9 * len(all_dataset))
val_size = len(all_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    all_dataset, [train_size, val_size]
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=settings.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    # timeout=60,
    collate_fn=lambda x: tuple(zip(*x)),
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=settings.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    # timeout=60,
    collate_fn=lambda x: tuple(zip(*x)),
)


inference_dataset = InferenceDataset(
    DATA_DIR_PATH / "test_release",
    DATA_DIR_PATH / "test_image_name_to_ids.json",
    transform=transform,
)
inference_loader = torch.utils.data.DataLoader(
    inference_dataset,
    batch_size=settings.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    # timeout=60,
    collate_fn=lambda x: tuple(zip(*x)),
)


def main():
    pass


if __name__ == "__main__":
    main()
