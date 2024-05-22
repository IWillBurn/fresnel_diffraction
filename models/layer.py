def layer_to_layer(layer):

    l = Layer(None, None, None, None)

    l.x = layer.x
    l.y = layer.y
    l.z = layer.z
    l.v = layer.v
    l.range = layer.range
    l.dt = layer.dt

    return l

class Layer:
    def __init__(self, x, y, z, v):
        self.x = x
        self.y = y
        self.z = z
        self.v = v
        self.range = None
        self.dt = None