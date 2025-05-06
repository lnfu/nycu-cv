import torch
import torch.nn as nn
import torch.nn.functional as F
from nycu_cv_hw3.constants import NUM_CLASSES
from torchvision.models.detection import (  # maskrcnn_resnet50_fpn_v2
    MaskRCNN_ResNet50_FPN_Weights,
    faster_rcnn,
    mask_rcnn,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import TwoMLPHead

# from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.ops import MultiScaleRoIAlign


class FourMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.fc8 = nn.Linear(representation_size, representation_size)
        self.fc9 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.maskrcnn = maskrcnn_resnet50_fpn(
            weights=weights,
            # 5 meaning all backbone layers are trainable
            trainable_backbone_layers=5,
            # num_classes=NUM_CLASSES,
            box_batch_size_per_image=800,
        )

        # transform 不要做 normalize
        self.maskrcnn.transform.image_mean = [0.0, 0.0, 0.0]
        self.maskrcnn.transform.image_std = [1.0, 1.0, 1.0]

        # unfreeze backbone
        # for param in self.maskrcnn.backbone.parameters():
        #     param.requires_grad = True

        # RoIHeads
        self.maskrcnn.roi_heads.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=(14, 14),  # original: (7, 7)
            sampling_ratio=2,
        )
        self.maskrcnn.roi_heads.box_head = TwoMLPHead(
            256 * 14 * 14, 1024  # 256 * 7 * 7 = 12544
        )
        # self.maskrcnn.roi_heads.box_head = FourMLPHead(
        #     256 * 14 * 14, 1024  # 256 * 7 * 7 = 12544
        # )

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

        # mask head
        # self.maskrcnn.roi_heads.mask_head = MaskRCNNHeads(
        #     self.maskrcnn.backbone.out_channels,
        #     [256, 256, 256, 256, 256, 256, 256, 256],
        #     dilation=1,
        # )

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


def main():
    model = Model()
    print(model)
    for name, param in model.maskrcnn.named_parameters():
        if not param.requires_grad:
            print(name)


if __name__ == "__main__":
    main()
