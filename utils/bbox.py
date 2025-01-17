import torch


def xywh2xyxy(bboxes):
    """
    Converts BBox of format cx,cy,w,h to x_min,y_min,x_max_y_max
    Args:
        bbox: (Tensor, float), shape (..., n, 4)
    Ref: https://github.com/pytorch/vision/blob/main/torchvision/ops/_box_convert.py
    """
    cx, cy, w, h = bboxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    bboxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return bboxes


def xyxy2xywh(bboxes):
    """
    Converts BBox of format x_min,y_min,x_max_y_max to cx,cy,w,h
    Args:
        bbox: (Tensor, float), shape (..., n, 4)
    Ref: https://github.com/pytorch/vision/blob/main/torchvision/ops/_box_convert.py
    """
    x1, y1, x2, y2 = bboxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    bboxes = torch.stack((cx, cy, w, h), dim=-1)
    return bboxes


if __name__ == "__main__":
    x = torch.full((1, 10, 4), 0.5, requires_grad=True)
    print(f"X {x}")
    x = xywh2xyxy(x)
    print(f"Y {x}")
    x = xyxy2xywh(x)
    print(f"Z {x}")
    x.mean().backward()
