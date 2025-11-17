import numpy as np

def compute_iou(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    ious = []
    for pb in pred_boxes:
        x1, y1, x2, y2 = pb
        for gb in gt_boxes:
            gx1, gy1, gx2, gy2 = gb
            inter_x1 = max(x1, gx1)
            inter_y1 = max(y1, gy1)
            inter_x2 = min(x2, gx2)
            inter_y2 = min(y2, gy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = (x2 - x1)*(y2 - y1) + (gx2 - gx1)*(gy2 - gy1) - inter_area
            if union_area > 0:
                ious.append(inter_area / union_area)
    return np.mean(ious) if ious else 0.0
