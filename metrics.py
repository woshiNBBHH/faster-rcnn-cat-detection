import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


class DetectionMetrics:
    """mAP plus threshold-based TP/FP/FN, over-detection and miss rates."""

    def __init__(self, score_threshold=0.5, iou_threshold=0.5):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, predictions, targets):
        cpu_predictions = [{key: value.detach().cpu() for key, value in pred.items()} for pred in predictions]
        cpu_targets = [{key: value.detach().cpu() for key, value in target.items()} for target in targets]
        self.map_metric.update(cpu_predictions, cpu_targets)

        for prediction, target in zip(cpu_predictions, cpu_targets):
            keep = prediction["scores"] >= self.score_threshold
            pred_boxes = prediction["boxes"][keep]
            pred_scores = prediction["scores"][keep]
            pred_labels = prediction["labels"][keep]
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            order = pred_scores.argsort(descending=True)
            matched_gt = set()
            for pred_index in order.tolist():
                same_class = torch.where(gt_labels == pred_labels[pred_index])[0]
                available = [idx.item() for idx in same_class if idx.item() not in matched_gt]
                if not available:
                    self.fp += 1
                    continue
                ious = box_iou(pred_boxes[pred_index].unsqueeze(0), gt_boxes[available])[0]
                best_value, best_offset = ious.max(dim=0)
                if best_value.item() >= self.iou_threshold:
                    matched_gt.add(available[best_offset.item()])
                    self.tp += 1
                else:
                    self.fp += 1
            self.fn += len(gt_boxes) - len(matched_gt)

    def compute(self):
        result = self.map_metric.compute()
        values = {key: float(value) for key, value in result.items() if value.numel() == 1}
        precision_denominator = self.tp + self.fp
        recall_denominator = self.tp + self.fn
        values.update({
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.tp / precision_denominator if precision_denominator else 0.0,
            "recall": self.tp / recall_denominator if recall_denominator else 0.0,
            "over_detection_rate": self.fp / precision_denominator if precision_denominator else 0.0,
            "miss_rate": self.fn / recall_denominator if recall_denominator else 0.0,
        })
        return values

