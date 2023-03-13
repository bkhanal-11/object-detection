import torch

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    BATCH_SIZE, num_anchors = predictions.shape[:2]
    box_predictions = predictions[..., 1:5]

    if is_preds:
        anchors = anchors.reshape(1, num_anchors, 1, 1, 2)
        box_predictions[..., :2] = torch.sigmoid(box_predictions[..., :2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., :1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1, keepdim=True)
    else:
        scores = predictions[..., :1]
        best_class = predictions[..., 5:6]

    cell_indices = torch.arange(S, device=predictions.device)
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices.view(1, 1, 1, S).float())
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.view(1, 1, S, 1).float())
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).view(BATCH_SIZE, num_anchors * S * S, 6)
    
    return converted_bboxes.tolist()
