from cmath import pi, exp

import torch

from config import CONFIG
from utils.metrics import torch_l2


def get_point_screen(center, point_val, z, output_layer):
    k = 2 * pi / CONFIG["lmbd"]
    kf = exp(complex(0, 1) * k * z) / (complex(0, 1) * CONFIG["lmbd"] * z)
    mult = torch.exp(torch.mul(torch.square(torch_l2(output_layer.x, output_layer.y, center[0], center[1])),
                               (complex(0, 1) * k / 2 / z)))
    ind = point_val
    output_layer.v = torch.mul(torch.mul(kf, mult), ind)

    return output_layer
