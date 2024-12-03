import torch

from typing import FrozenSet, Generic, Union

from chebai_towns.towns.shape import FrameShape, Point, Box, _All, _Nothing, ALL, NOTHING
from matplotlib import pyplot as plt
import numpy as np
import shapely


class House:
    frame: Union[FrameShape, _All]
    windows: FrozenSet[FrameShape]

    def __init__(self, frame, windows):
        self.frame = frame
        self.windows = windows

    def __neg__(self) -> "Town":
        houses = ([House(ALL, [self.frame])]
                  + [House(window, NOTHING) for window in self.windows])
        return Town(
            houses = houses
        )

    def __and__(self, other):
        return House(self.frame and other.frame, self.windows.union(other.windows))

    def contains_points(self, points):
        containments = self.detailed_containment(points)
        return containments["frame_containment"] * (1-torch.max(containments["window_containments"], dim=0)[0])

    def fuzzy_contains_points(self, points):
        containments = self.detailed_containment(points)
        return containments["frame_containment"] * (1-torch.max(containments["window_containments"], dim=0)[0])

    def detailed_containment(self, points):
        frame_containment = self.frame.contains_points(points)
        window_containment = torch.stack([(self.frame.intersection(w)).contains_points(points) for w in self.windows], dim=0)
        return dict(frame_containment=frame_containment, window_containments=window_containment)

    def detailed_distances(self, points, p=2):
        frame_distance = self.frame.distance_to_points(points, p=p)
        intersections = [w for w in self.windows if w != NOTHING]
        inner_window_distances = None
        outer_window_distances = None
        center_window_distances = None
        if intersections:
            distances = [w.distance_to_points(points, p=p) for w in intersections]
            inner_window_distances = torch.stack([d["inside"] for d in distances], dim=0)
            outer_window_distances = torch.stack([d["outside"] for d in distances], dim=0)
            center_window_distances = torch.stack([d["center"] for d in distances], dim=0)
        return dict(inner_frame_distance=frame_distance["inside"], outer_frame_distance=frame_distance["outside"], center_frame_distance=frame_distance["center"], inner_window_distances=inner_window_distances, outer_window_distances=outer_window_distances, center_window_distances=center_window_distances)

    def render(self, ax: plt.Axes, color):
        window_patches = [w.get_patch() for w in self.windows]
        window_union = shapely.union_all(window_patches)
        shape = shapely.difference(self.frame.get_patch(), window_union)
        if not shapely.is_empty(shape):
            if isinstance(shape, shapely.MultiPolygon):
                for geom in shape.geoms:
                    xs, ys = geom.exterior.xy
                    ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)
            else:
                xs, ys = shape.exterior.xy
                ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)

    def consolidate(self):
        self.windows = [self.frame.intersection(w) for w in self.windows]

class Town:
    houses: FrozenSet[House]

    def __init__(self, houses):
        self.houses = houses
        self.color = list(np.random.choice(range(256), size=3)/255)

    def __and__(self, other):
        return Town(frozenset(h for a in other.houses for b in other.houses for h in (a and b)))

    def __or__(self, other: "Town"):
        return Town(self.houses.union(other.houses))

    def __neg__(self):
        if not self.houses:
            return ALL
        else:
            neg_town = None
            for house in self.houses:
                if neg_town is None:
                    neg_town = not house
                else:
                    neg_town = neg_town and not house
        return neg_town

    def detailed_distances(self, points):
        return [h.detailed_distances(points) for h in self.houses]

    def contains_points(self, points):
        return torch.max(torch.stack([h.contains_points(points) for h in self.houses], dim=0), dim=0)[0]

    def fuzzy_contains_points(self, points):
        return torch.max(torch.stack([h.fuzzy_contains_points(points) for h in self.houses], dim=0), dim=0)[0]

    def render(self, ax: plt.Axes):
        for h in self.houses:
            h.render(ax, self.color)

    def consolidate(self):
        for h in self.houses:
            h.consolidate()


if __name__=="__main__":
    from matplotlib import pyplot as plt
    frame = Box(torch.tensor(((10, 20), (30, 40))))
    window_1 = Box(torch.tensor(((12, 22), (16, 24))))
    window_2 = Box(torch.tensor(((13, 23), (45, 40))))
    house = House(frame, [window_1, window_2])
    ranges = torch.linspace(0, 50, 500)
    a, b = torch.meshgrid(ranges, ranges, indexing="xy")
    distance = house.distance_to_points(torch.stack((a, torch.flip(b, dims=(0,))), dim=-1))
    ax = plt.gca()
    plt.imshow(distance, cmap='hot', interpolation='nearest', extent=(0, 50, 0, 50))
    house.render(ax, (0,1,0))
    plt.show()