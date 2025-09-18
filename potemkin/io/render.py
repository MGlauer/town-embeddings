from matplotlib import pyplot as plt
from matplotlib.artist import Artist

def draw_2d(points, town_model=None, point_colors = None, town_colors = None, xlim=None, ylim=None):
    plt.clf()
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    draw_2d_on_axes(points, ax, town_model, point_colors, town_colors)
    return fig

def draw_2d_on_axes(points, ax, town_model=None, point_colors = None, town_colors = None) -> list[Artist]:
    artists: list[Artist] = []
    artists.append(
        ax.scatter(points[:, 0], points[:, 1], marker=".", c=point_colors)
    )
    if town_model is not None:
        artists += town_model.box_model.render(ax, town_colors)
    return artists