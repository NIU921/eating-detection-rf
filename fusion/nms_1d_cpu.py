"""Portable CPU fallback for ActionFormer's 1D NMS extension."""
import math

import torch


def nms(segs, scores, iou_threshold):
    boxes = segs.detach().cpu().numpy()
    order = scores.argsort(descending=True).tolist()
    keep = []
    for i in order:
        a, b = boxes[i]
        matched = False
        for j in keep:
            c, d = boxes[j]
            inter = max(0.0, min(b, d) - max(a, c))
            overlap = inter / max(1e-6, (b - a + 1e-6) + (d - c + 1e-6) - inter)
            if overlap >= iou_threshold:
                matched = True
                break
        if not matched:
            keep.append(i)
    return torch.tensor(keep, dtype=torch.long)


def softnms(segs, scores, dets, iou_threshold, sigma, min_score, method):
    boxes = segs.detach().cpu().numpy().copy()
    values = scores.detach().cpu().numpy().copy()
    output = dets.detach().cpu().numpy()
    indices = list(range(len(values)))
    count = len(values)
    i = 0
    while i < count:
        best = i + int(values[i:count].argmax())
        boxes[[i, best]] = boxes[[best, i]]
        values[[i, best]] = values[[best, i]]
        indices[i], indices[best] = indices[best], indices[i]
        a, b = boxes[i]
        output[i] = (a, b, values[i])
        j = i + 1
        while j < count:
            c, d = boxes[j]
            inter = max(0.0, min(b, d) - max(a, c))
            overlap = inter / max(1e-6, (b - a + 1e-6) + (d - c + 1e-6) - inter)
            if method == 0:
                weight = 0.0 if overlap >= iou_threshold else 1.0
            elif method == 1:
                weight = 1.0 - overlap if overlap >= iou_threshold else 1.0
            else:
                weight = math.exp(-(overlap * overlap) / sigma)
            values[j] *= weight
            if values[j] < min_score:
                boxes[j] = boxes[count - 1]
                values[j] = values[count - 1]
                indices[j] = indices[count - 1]
                count -= 1
            else:
                j += 1
        i += 1
    return torch.tensor(indices[:count], dtype=torch.long)
