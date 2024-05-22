import torch


class RadiusBasedIndicator:
    def __init__(self, metric, r, center, reverse):
        self.metric = metric
        self.r = r
        self.center = center
        self.reverse = reverse

    def calculate(self, x, y):
        if not self.reverse:
            return torch.where(self.metric(self.center[0], self.center[1], x, y) > self.r, 0, 1)

        return torch.where(self.metric(self.center[0], self.center[1], x, y) < self.r, 0, 1)
