import sys

import torch


def eprint(*args, **kwargs):
    """
    Print to stderr instead of stdout.
    """
    print(*args, file=sys.stderr, **kwargs)


def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    Compute the IoU (Intersection over Union) matrix between two sets of bounding boxes.

    Args:
        boxes1 (Tensor[N, 4]): Bounding boxes in (x1, y1, x2, y2) format.
        boxes2 (Tensor[M, 4]): Bounding boxes in (x1, y1, x2, y2) format.

    Returns:
        Tensor[N, M]: IoU matrix where element (i, j) is the IoU between boxes1[i] and boxes2[j].
    """
    x_min_inter = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y_min_inter = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x_max_inter = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y_max_inter = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    # intersection (不能小於 0)
    # inter_w = torch.maximum(torch.tensor(0.0), x_max_inter - x_min_inter)
    # inter_h = torch.maximum(torch.tensor(0.0), y_max_inter - y_min_inter)
    inter_w = (x_max_inter - x_min_inter).clamp(min=0)
    inter_h = (y_max_inter - y_min_inter).clamp(min=0)
    inter_area = inter_w * inter_h

    # area
    box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # union
    union_area = box1_area[:, None] + box2_area[None, :] - inter_area

    # iou_matrix = inter_area / union_area
    # return iou_matrix
    iou_matrix = torch.where(
        union_area > 0, inter_area / union_area, torch.zeros_like(union_area)
    )
    return iou_matrix
