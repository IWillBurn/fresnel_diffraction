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


def single_slit_experiment(r):

    # PIPELINE
    print("PIPELINE")
    pipeline = Pipeline()

    # STAGES
    print("STAGES")

    dt = 0.00001
    x, y = get_range([[-0.001, 0.001], [-0.001, 0.001]], dt)
    ind = RadiusBasedIndicator(torch_single, r, [0, 0], False)
    model = FresnelModel(ind.calculate)
    output_layer = Layer(x, y, 5, None)
    output_layer.range = [[-0.001, 0.001], [-0.001, 0.001]]
    output_layer.dt = dt
    stage2 = Stage(model, ind.calculate, output_layer)

    stages = [stage2]

    # INPUT LAYER

    dt = 0.00001
    x, y = get_range([[-0.001, 0.001], [-0.001, 0.001]], dt)
    print("INPUT LAYER")
    intens = 1
    input_layer = Layer(x, y, 0, torch.Tensor([intens for i in range(len(x))]))
    input_layer.range = [[-0.001, 0.001], [-0.001, 0.001]]
    input_layer.dt = dt

    # PIPELINE WORK
    print("PIPELINE WORK")
    result = pipeline.work(input_layer, stages)

    draw_layer_formal(result, "result")
    get_graph(result)