import torch

from draw.drawer import draw_layer_formal
from integral_models.fresnel import FresnelModel
from draw.utils import get_graph
from indicators.indicators import RadiusBasedIndicator
from models.layer import Layer
from utils.metrics import torch_l2
from pipelines.pipeline import Pipeline
from utils.ranges import get_range
from models.stage import Stage


def argo_experiment(r, d):

    # PIPELINE
    print("PIPELINE")
    pipeline = Pipeline()

    # STAGES
    print("STAGES")

    dt = 0.00005
    x, y = get_range([[-d / 2, d / 2], [-d / 2, d / 2]], dt)
    ind = RadiusBasedIndicator(torch_l2, r, [0, 0], True)
    model = FresnelModel(ind.calculate)
    output_layer = Layer(x, y, 1, None)
    output_layer.range = [[-d / 2, d / 2], [-d / 2, d / 2]]
    output_layer.dt = dt
    stage2 = Stage(model, ind.calculate, output_layer)

    stages = [stage2]

    # INPUT LAYER

    dt = 0.00003
    x, y = get_range([[-d / 2, d / 2], [-d / 2, d / 2]], dt)
    print("INPUT LAYER")
    intens = 1
    input_layer = Layer(x, y, 0, torch.Tensor([intens for i in range(len(x))]))
    input_layer.range = [[-d / 2, d / 2], [-d / 2, d / 2]]
    input_layer.dt = dt

    # PIPELINE WORK
    print("PIPELINE WORK")
    result = pipeline.work(input_layer, stages)

    draw_layer_formal(result, "result")
    get_graph(result)