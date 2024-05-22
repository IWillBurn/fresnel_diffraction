import torch

from draw.drawer import draw_layer_formal
from integral_models.fresnel import FresnelModel
from draw.utils import get_graph
from indicators.indicators import RadiusBasedIndicator
from models.layer import Layer
from utils.metrics import torch_single
from pipelines.pipeline import Pipeline
from utils.ranges import get_range
from models.stage import Stage


def life_experiment(r):

    # PIPELINE
    print("PIPELINE")
    pipeline = Pipeline()

    # STAGES
    print("STAGES")

    dt = 0.00001
    x, y = get_range([[-0.001, 0.001], [-0.001, 0.001]], dt)
    ind = RadiusBasedIndicator(torch_single, 0.001, [0, 0], True)
    model = FresnelModel(ind.calculate)
    output_layer = Layer(x, y, 2, None)
    output_layer.range = [[-0.001, 0.001], [-0.001, 0.001]]
    output_layer.dt = dt
    stage1 = Stage(model, ind.calculate, output_layer)

    dt = 0.00005
    x, y = get_range([[-0.005, 0.005], [-0.005, 0.005]], dt)
    ind = RadiusBasedIndicator(torch_single, 0.0005, [0, 0], True)
    model = FresnelModel(ind.calculate)
    output_layer = Layer(x, y, 4, None)
    output_layer.range = [[-0.005, 0.005], [-0.005, 0.005]]
    output_layer.dt = dt
    stage2 = Stage(model, ind.calculate, output_layer)

    stages = [stage1, stage2]

    # INPUT LAYER

    dt = 0.00005
    x, y = get_range([[-0.005, 0.005], [-0.005, 0.005]], dt)
    print("INPUT LAYER")
    intens = 10
    input_layer = Layer(x, y, 0, torch.Tensor([intens for i in range(len(x))]))
    input_layer.range = [[-0.005, 0.005], [-0.005, 0.005]]
    input_layer.dt = dt

    # PIPELINE WORK
    print("PIPELINE WORK")
    result = pipeline.work(input_layer, stages)

    draw_layer_formal(result, "result")
    get_graph(result)