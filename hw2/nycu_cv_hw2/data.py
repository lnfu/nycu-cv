import logging
import pathlib

import torch
import torchvision
from nycu_cv_hw2.constants import DATA_DIR_PATH

# Dataset
train_dataset = torchvision.datasets.CocoDetection(
    root=DATA_DIR_PATH / "train",
    annFile=DATA_DIR_PATH / "train.json",
    transform=torchvision.transforms.transforms.ToTensor(),  # TODO
    target_transform=None,
)
val_dataset = torchvision.datasets.CocoDetection(
    root=DATA_DIR_PATH / "valid",
    annFile=DATA_DIR_PATH / "valid.json",
    transform=torchvision.transforms.transforms.ToTensor(),  # TODO
    target_transform=None,
)


# DataLoader
def collate_fn(batch):
    inputs, labels = zip(*batch)

    new_labels = []
    for label in labels:  # label is list of dicts
        boxes = []
        # bbox = (x_min, y_min, width, height) -> bbox = (x_min, y_min, x_max, y_max)
        for t in label:
            x_min, y_min, w, h = t["bbox"]
            x_max = x_min + w
            y_max = y_min + h
            if w <= 0 or h <= 0:
                logging.warning(f"Found invalid box {t['bbox']} -> Skipping.")
                continue
            boxes.append([x_min, y_min, x_max, y_max])

        new_label = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(
                [t["category_id"] - 1 for t in label],  # category_id =  1-10 -> 0-9
                dtype=torch.int64,
            ),
            "image_id": torch.tensor([label[0]["image_id"]], dtype=torch.int64),
            "area": torch.tensor([t["area"] for t in label], dtype=torch.float32),
            "iscrowd": torch.tensor([t["iscrowd"] for t in label], dtype=torch.int64),
        }
        new_labels.append(new_label)

    return list(inputs), new_labels


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)
