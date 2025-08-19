from matplotlib import pyplot as plt

def draw_2d(points, town_model=None, point_colors = None, town_colors = None):
    plt.clf()
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], marker=".", c=point_colors)
    if town_model is not None:
        town_model.box_model.render(ax, town_colors)
    return fig
