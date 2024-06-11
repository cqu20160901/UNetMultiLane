import torch


def softmax_focal_loss(inputs_seg_cls, seg_targets, cls_targets, gamma=2, alpha=0.25):
    # seg
    inputs = inputs_seg_cls[0]
    inputs = torch.nn.functional.softmax(inputs, dim=1)
    inputs = torch.clamp(inputs, min=1e-3, max=1 - 1e-3)
    target_one_hot = torch.zeros_like(inputs).scatter_(1, seg_targets, 1)

    pos = alpha * ((1 - inputs) ** gamma) * torch.log(inputs)
    neg = (1 - alpha) * (inputs ** gamma) * torch.log(1 - inputs)
    pos = target_one_hot * pos
    neg = -(target_one_hot - 1) * neg
    seg_loss = torch.mean(pos + neg)

    # cls
    cls_inputs = inputs_seg_cls[1]
    cls_inputs = torch.nn.functional.softmax(cls_inputs, dim=2)
    cls_inputs = torch.clamp(cls_inputs, min=1e-3, max=1 - 1e-3)
    cls_label = cls_targets.unsqueeze(2)
    cls_one_hot = torch.zeros_like(cls_inputs).scatter_(2, cls_label, 1)

    pos1 = alpha * ((1 - cls_inputs) ** gamma) * torch.log(cls_inputs)
    neg1 = (1 - alpha) * (cls_inputs ** gamma) * torch.log(1 - cls_inputs)
    pos1 = cls_one_hot * pos1
    neg1 = -(cls_one_hot - 1) * neg1
    cls_loss = torch.mean(pos1 + neg1)

    return -seg_loss, -cls_loss
