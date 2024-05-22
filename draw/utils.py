from math import floor

from matplotlib import pyplot as plt


def get_graph(layer):
    x_start = (floor((layer.range[0][1] - layer.range[0][0]) / layer.dt) + 1) // 2
    y_size = floor((layer.range[0][1] - layer.range[0][0]) / layer.dt) + 1

    xs = [layer.range[0][0] + i * layer.dt for i in range(floor((layer.range[0][1] - layer.range[0][0]) / layer.dt) + 1)]
    ys = layer.v[x_start::y_size]

    ys = [abs(i.item()) ** 2 for i in ys]

    plt.figure()
    plt.plot(xs, ys, 'g.', label='Точки')
    plt.plot(xs, ys, 'b-', label='Точки')
    plt.xlabel('X ось')
    plt.ylabel('Y ось')
    plt.title('График по точкам')
    plt.legend()
    plt.grid(True)

    # отображаем график
    plt.show()