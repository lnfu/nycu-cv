import sys

import torch


def eprint(*args, **kwargs):
    """TODO"""
    print(*args, file=sys.stderr, **kwargs)


def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """TODO"""
    x_min_intersection = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y_min_intersection = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x_max_intersection = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y_max_intersection = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    # intersection (不能小於 0)
    intersection_width = torch.maximum(
        torch.tensor(0.0), x_max_intersection - x_min_intersection
    )
    intersection_height = torch.maximum(
        torch.tensor(0.0), y_max_intersection - y_min_intersection
    )
    intersection_area = intersection_width * intersection_height

    # area
    box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # union
    union_area = box1_area[:, None] + box2_area[None, :] - intersection_area

    iou_matrix = intersection_area / union_area
    return iou_matrix
