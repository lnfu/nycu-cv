import torch
import torch.nn as nn
from nycu_cv_hw3.constants import NUM_CLASSES
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    faster_rcnn,
    mask_rcnn,
    maskrcnn_resnet50_fpn,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.maskrcnn = maskrcnn_resnet50_fpn(weights=weights)

        # class predictor
        in_features_box = (
            self.maskrcnn.roi_heads.box_predictor.cls_score.in_features
        )
        self.maskrcnn.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
            in_features_box, NUM_CLASSES
        )

        # mask predictor
        in_features_mask = (
            self.maskrcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        )
        self.maskrcnn.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(
            in_features_mask, 256, NUM_CLASSES
        )
        # print(self.maskrcnn)
        n_parameters = sum(p.numel() for p in self.maskrcnn.parameters())
        print(f"{n_parameters=}")
        assert n_parameters <= 200 * 1024 * 1024

    def forward(self, *args):
        return self.maskrcnn(*args)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: torch.device):
        model = cls()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
