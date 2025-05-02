import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
import torch
from nycu_cv_hw3.config import settings
from nycu_cv_hw3.constants import DATA_DIR_PATH
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.ops import masks_to_boxes


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

        if self.transform:
            image = self.transform(image)

        labels = []
        masks = []

        for path in sample_dir.glob("class*.tif"):
            # e.g. class3.tif -> 3
            label = int(path.stem.replace("class", ""))

            mask = tifffile.imread(path).astype(np.uint8)  # H x W
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]
            for instance_id in instance_ids:
                labels.append(label)
                masks.append((mask == instance_id))

        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        boxes = masks_to_boxes(masks)
        # area = masks.sum(dim=[1, 2]) if len(masks) > 0 else torch.tensor([])
        # iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        image_id = torch.tensor([idx])

        return image, {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            # "area": area,
            # "iscrowd": iscrowd,
            "image_id": image_id,
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
            image = self.transform(image)

        return (
            image,
            image_name,
            self.metadata[image_name]["id"],
            self.metadata[image_name]["width"],
            self.metadata[image_name]["height"],
        )


transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 0–255 → 0.0–1.0
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

all_dataset = TrainDataset(DATA_DIR_PATH / "train", transform=transform)

# TODO config
train_size = int(0.8 * len(all_dataset))
val_size = len(all_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    all_dataset, [train_size, val_size]
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=settings.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda x: tuple(zip(*x)),
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=settings.batch_size,
    shuffle=False,
    num_workers=4,
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
    num_workers=4,
    collate_fn=lambda x: tuple(zip(*x)),
)
