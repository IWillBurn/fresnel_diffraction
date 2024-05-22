import torch


def torch_l2(x1, y1, x2, y2):
    return torch.sqrt(torch.add(torch.square(torch.subtract(x2, x1)), torch.square(torch.subtract(y2, y1))))


def torch_double(x1, y1, x2, y2):
    return torch.max(torch.min(torch.abs(torch.subtract(x2, x1)), torch.abs(torch.subtract(x2, torch.mul(x1, -1)))), torch.mul(torch.abs(torch.subtract(y2, y1)), 3))


def torch_single(x1, y1, x2, y2):
    return torch.max(torch.abs(torch.subtract(x2, x1)), torch.mul(torch.abs(torch.subtract(y2, y1)), 3))
