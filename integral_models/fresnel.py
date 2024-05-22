from cmath import exp, pi

import torch

from config import CONFIG
from utils.metrics import torch_l2


class FresnelModel:
    def __init__(self, ind_func):
        self.ind_func = ind_func

    def func(self, x, y, z, px, py):
        k = 2 * pi / CONFIG["lmbd"]
        kf = exp(complex(0, 1) * k * z) / (complex(0, 1) * CONFIG["lmbd"] * z)
        mult = torch.exp(torch.mul(torch.square(torch_l2(px, py, x, y)), (complex(0, 1) * k / 2 / z)))
        ind = self.ind_func(px, py)
        return torch.mul(torch.mul(kf, mult), ind)

    def func_without_ind(self, x, y, z, px, py, c):
        k = 2 * pi / CONFIG["lmbd"]
        kf = exp(complex(0, 1) * k * z) / (complex(0, 1) * CONFIG["lmbd"] * z)
        mult = torch.exp(torch.mul(torch.square(torch_l2(px, py, x, y)), (complex(0, 1) * k / 2 / z)))
        ind = c
        return torch.mul(torch.mul(kf, mult), ind)