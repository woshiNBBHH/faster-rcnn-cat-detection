from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes, pretrained=True, trainable_backbone_layers=3,
                min_size=640, max_size=1024, nms_threshold=0.5):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    effective_trainable_layers = trainable_backbone_layers if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=effective_trainable_layers,
        min_size=min_size,
        max_size=max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = nms_threshold
    return model
