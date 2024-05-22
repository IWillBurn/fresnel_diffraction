from math import floor

import torch


class ModelHolder:
    def __init__(self, model):
        self.model = model
        self.x = 0
        self.y = 0
        self.z = 0

    def set_point(self, point):
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]

    def calculate(self, inp):
        x = inp[:, 0]
        y = inp[:, 1]
        return self.model.func(self.x, self.y, self.z, x, y)


class CustomModelHolder:
    def __init__(self, model):
        self.model = model
        self.x = 0
        self.y = 0
        self.z = 0
        self.input_layer = None
        self.inp_selections_x = None
        self.inp_selections_y = None
        self.inp_selections_v = None

    def set_point(self, point):
        self.x = point[0]
        self.y = point[1]
        self.z = point[2]

    def calculate(self, inp):
        x = inp[:, 0]
        y = inp[:, 1]
        return self.model.func_without_ind(self.x, self.y, self.z, self.inp_selections_x, self.inp_selections_y, self.inp_selections_v)

    def calc_sel(self, n):
        self.inp_selections_x = self.input_layer.x
        self.inp_selections_y = self.input_layer.y

        self.inp_selections_v = self.input_layer.v