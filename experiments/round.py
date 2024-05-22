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


def round_experiment(r):

    # PIPELINE
    print("PIPELINE")
    pipeline = Pipeline()

    # STAGES
    print("STAGES")

    dt = 0.00003
    x, y = get_range([[-0.005, 0.005], [-0.005, 0.005]], dt)
    ind = RadiusBasedIndicator(torch_l2, 0.0001, [0, 0], False)
    model = FresnelModel(ind.calculate)
    output_layer = Layer(x, y, 2, None)
    output_layer.range = [[-0.005, 0.005], [-0.005, 0.005]]
    output_layer.dt = dt
    stage1 = Stage(model, ind.calculate, output_layer)

    dt = 0.00002
    x, y = get_range([[-0.002, 0.002], [-0.002, 0.002]], dt)
    ind = RadiusBasedIndicator(torch_l2, r, [0, 0], False)
    model = FresnelModel(ind.calculate)
    output_layer = Layer(x, y, 3, None)
    output_layer.range = [[-0.002, 0.002], [-0.002, 0.002]]
    output_layer.dt = dt
    stage2 = Stage(model, ind.calculate, output_layer)

    stages = [stage1, stage2]

    # INPUT LAYER

    dt = 0.000002
    x, y = get_range([[-0.0002, 0.0002], [-0.0002, 0.0002]], dt)
    print("INPUT LAYER")
    intens = 10
    input_layer = Layer(x, y, 0, torch.Tensor([intens for i in range(len(x))]))
    input_layer.range = [[-0.0002, 0.0002], [-0.0002, 0.0002]]
    input_layer.dt = dt

    # PIPELINE WORK
    print("PIPELINE WORK")
    result = pipeline.work(input_layer, stages)

    draw_layer_formal(result, "result")
    get_graph(result)