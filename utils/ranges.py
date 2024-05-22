from math import floor

import torch


def get_range(screen_domain, dt):

    x = torch.FloatTensor(
        [screen_domain[0][0] + i * dt for i in range(floor((screen_domain[0][1] - screen_domain[0][0]) / dt) + 1) for j
         in range(floor((screen_domain[1][1] - screen_domain[1][0]) / dt) + 1)])

    y = torch.FloatTensor(
        [screen_domain[1][0] + j * dt for i in range(floor((screen_domain[0][1] - screen_domain[0][0]) / dt) + 1) for j
         in range(floor((screen_domain[1][1] - screen_domain[1][0]) / dt) + 1)])

    return x, y