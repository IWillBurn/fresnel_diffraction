"""

stage {
    cover_func
    output_layer
    model
}

"""
import numpy as np
import torch
from torchquad import MonteCarlo
from tqdm import tqdm

from draw.drawer import draw_layer_formal
from models.holder import CustomModelHolder
from models.layer import Layer, layer_to_layer


class Pipeline:
    def __init__(self):
        pass

    def work(self, input_layer, stages):
        for i, stage in enumerate(stages):
            print(f"stage {i+1}/{len(stages)}:")
            input_layer = self.apply_stage(input_layer, stage)
            draw_layer_formal(input_layer, i+1)

        return input_layer

    def apply_stage(self, input_layer, stage):

        cover_layer = Layer(input_layer.x, input_layer.y, 0, None)

        cover_layer.v = stage.cover_func(cover_layer.x, cover_layer.y)
        draw_layer_formal(cover_layer, "cover")

        input_layer.v = torch.mul(input_layer.v, cover_layer.v)

        h = CustomModelHolder(stage.model)
        h.input_layer = input_layer

        mc = MonteCarlo()
        result = []

        h.calc_sel(100000)

        for i in tqdm(range(len(stage.output_layer.x))):
            h.set_point([stage.output_layer.x[i], stage.output_layer.y[i], stage.output_layer.z - input_layer.z])
            value = mc.integrate(h.calculate, 2, N=len(input_layer.x), integration_domain=input_layer.range)
            result.append(value)

        del h
        del mc

        result_layer = layer_to_layer(stage.output_layer)
        result = [i.item() for i in result]
        result_layer.v = torch.from_numpy(np.array(result))
        return result_layer




