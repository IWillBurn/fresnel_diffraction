from cmath import phase, pi

from matplotlib import pyplot as plt

def draw_layer_formal(layer, num):
    col = []

    for i in layer.v:
        if i != 0:
            col.append(abs(i) ** 2) # abs(i) ** 2
        else:
            col.append(0)

    plt.scatter(layer.x, layer.y, c=col, cmap='gray', s=1)  # Используем серую цветовую карту

    plt.title(f'График с яркостью точек. Слой {num}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Яркость')

    plt.show()