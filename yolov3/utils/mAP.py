from collections import Counter
import torch

from .iou import intersection_over_union

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    epsilon = 1e-6
    average_precisions = []

    for c in range(num_classes):
        detections = [d for d in pred_boxes if d[1] == c]
        ground_truths = [t for t in true_boxes if t[1] == c]
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        detections.sort(key=lambda x: x[2], reverse=True)
        TP, FP = torch.zeros((len(detections))), torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        if total_true_bboxes == 0: continue
        
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts, best_iou, best_gt_idx = len(ground_truth_img), 0, None
            
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)
                
                if iou > best_iou: best_iou, best_gt_idx = iou, idx
            
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0: TP[detection_idx], amount_bboxes[detection[0]][best_gt_idx] = 1, 1
                else: FP[detection_idx] = 1
            else: FP[detection_idx] = 1
        
        TP_cumsum, FP_cumsum = torch.cumsum(TP, dim=0), torch.cumsum(FP, dim=0)
        recalls, precisions = TP_cumsum / (total_true_bboxes + epsilon), TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions) / len(average_precisions)
