from typing import Dict, Any, Iterable

from potemkin.towns import shape
from potemkin.towns.shape import FrameShape, Box
from potemkin.towns.geometry import House, Town
import torch
from potemkin.towns.shape import Box
import shapely

class GeometryNet(torch.nn.Module):

    def __init__(self, embedding_dimensions: int, num_classes, num_houses_per_class=5, num_windows_per_house=4,
                 shape_kwargs=None):
        super().__init__()

        self.town_frame_tensor = torch.nn.Parameter(3-6*torch.rand((num_classes, num_houses_per_class, 2, embedding_dimensions)))
        self.town_windows_tensor = torch.nn.Parameter(3-6*torch.rand((num_classes, num_houses_per_class, num_windows_per_house, 2, embedding_dimensions)))
        self.p = 2

    def forward(self, points):
        points_2 = points.unsqueeze(1).unsqueeze(-2)
        min_frame = torch.min(self.town_frame_tensor, dim=-2)[0].unsqueeze(0)
        max_frame = torch.max(self.town_frame_tensor, dim=-2)[0].unsqueeze(0)

        min_windows = torch.min(self.town_windows_tensor, dim=-2)[0].unsqueeze(0)
        max_windows = torch.max(self.town_windows_tensor, dim=-2)[0].unsqueeze(0)

        frame_containment = self._inside(min_frame, max_frame, points_2)
        inner_frame_distance = self._inner_distance(min_frame, max_frame, points_2, p=self.p)
        outer_frame_distance = self._outer_distance(min_frame, max_frame, points_2, p=self.p)

        points_3 = points_2.unsqueeze(-2)
        window_containment = self._inside(min_windows, max_windows, points_3)
        inner_window_distance = self._inner_distance(min_windows, max_windows, points_3, p=self.p)
        outer_window_distance = self._outer_distance(min_windows, max_windows, points_3, p=self.p)

        house_containment = frame_containment * (1 - torch.max(window_containment, dim=-1)[0])
        containment = torch.max(house_containment, dim=-1)[0]

        return dict(
            embeddings=points,
            crisp_frame_containment=frame_containment,
            crisp_window_containment=window_containment,
            crisp_house_containment=house_containment,
            crisp_containment=containment,
            inner_frame_distance=inner_frame_distance,
            outer_frame_distance=outer_frame_distance,
            inner_window_distances=inner_window_distance,
            outer_window_distances=outer_window_distance,
        )

    def _inside(self, l, r, p):
        return torch.prod((l <= p) * (p <= r), dim=-1)

    def _outer_distance(self, min_corner, max_corner, point, p=2):
        margin = 0.1 * (max_corner - min_corner).detach()
        return self._norm(torch.relu(min_corner + margin - point) + torch.relu(point - max_corner + margin), dim=-1, p=p)

    def _inner_distance(self, min_corner, max_corner, point, p=2):
        margin = 0.1 * (max_corner - min_corner).detach()
        return self._norm(
            torch.minimum(torch.relu(max_corner + margin - point), torch.relu(point - min_corner + margin)), dim=-1, p=p)

    def _norm(self, point, p=2, dim=-1):
        if p is None:
            return torch.max(torch.abs(point), dim=dim)[0]
        else:
            return torch.sum(torch.abs(point)**p, dim=dim)**(1/p)

    def _tensor_to_patch(self, tensor):
        min_frame = torch.min(tensor, dim=-2)[0]
        max_frame = torch.max(tensor, dim=-2)[0]
        return shapely.box(min_frame[0].item(), min_frame[1].item(), max_frame[0].item(), max_frame[1].item())

    def get_patches(self) -> Iterable[shapely.Geometry]:
        towns = self.town_windows_tensor.shape[0]
        houses = self.town_windows_tensor.shape[1]
        windows = self.town_windows_tensor.shape[2]
        if towns == 1 and houses == 1 and windows == 1:
            return [self._tensor_to_patch(self.town_frame_tensor[0, 0]), self._tensor_to_patch(self.town_windows_tensor[0, 0, 0])]
        town_shapes = []
        for town in range(towns):
            house_shapes = []
            for house in range(houses):
                house_shape = self._tensor_to_patch(self.town_frame_tensor[town, house])
                for window in range(windows):
                    window_shape = self._tensor_to_patch(self.town_windows_tensor[town, house, window])
                    house_shape = shapely.difference(house_shape, window_shape)
                house_shapes.append(house_shape)
            town_shapes.append(shapely.union_all(house_shapes))
        return town_shapes

    def render(self, ax, colors):
        polygons = []
        for shape, color in zip(self.get_patches(), colors):
            if isinstance(shape, shapely.MultiPolygon):
                for geom in shape.geoms:
                    xs, ys = geom.exterior.xy
                    polygons += ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)
            else:
                xs, ys = shape.exterior.xy
                polygons += ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)
        return polygons

class TownModel(torch.nn.Module):
    def __init__(self, embedding_model, out_dim: int, num_houses_per_class=5, num_windows_per_house=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        self.embedding_model = embedding_model
        self.box_model = GeometryNet(embedding_dimensions=self.embedding_model.out_dim, num_classes=out_dim, num_houses_per_class=num_houses_per_class, num_windows_per_house=num_windows_per_house)

    def forward(self, x, **kwargs):
        embeddings = self.embedding_model(x, **kwargs)
        distances = self.box_model(embeddings)
        return distances
    
class ConeGeometryNet(torch.nn.Module):

    def __init__(self, embedding_dimensions: int, num_classes, num_houses_per_class=5, num_windows_per_house=4,
                 shape_kwargs=None):
        super().__init__()

        self.town_frame_tensor = torch.nn.Parameter(3-6*torch.rand((num_classes, num_houses_per_class, 2, embedding_dimensions)))
        self.town_windows_tensor = torch.nn.Parameter(3-6*torch.rand((num_classes, num_houses_per_class, num_windows_per_house, 2, embedding_dimensions)))
        self.p = 2

    def forward(self, points):
        # frame_containment.shape  == (n_points, n_labels, num_houses_per_class)
        frame_containment = self._inside(self.town_frame_tensor, points)
        # window_containment.shape == (n_points, n_labels, num_houses_per_class, num_windows_per_house)
        window_containment = self._inside(self.town_windows_tensor, points)
        # house_containment.shape  == (n_points, n_labels, num_houses_per_class)
        house_containment = frame_containment * (1 - torch.max(window_containment, dim=-1)[0])
        # containment.shape        == (n_points, n_labels)
        containment = torch.max(house_containment, dim=-1)[0]

        frame_distance = self._distance(self.town_frame_tensor, points, p=self.p)
        # inner_frame_distance.shape == (n_points, n_labels, num_houses_per_class)
        inner_frame_distance = frame_containment * frame_distance
        # outer_frame_distance.shape == (n_points, n_labels, num_houses_per_class)
        outer_frame_distance = (1 - frame_containment) * frame_distance

        window_distance = self._distance(self.town_windows_tensor, points, p=self.p)
        # inner_window_distance.shape == (n_points, n_labels, num_houses_per_class, num_windows_per_house)
        inner_window_distance = window_containment * window_distance
        # outer_window_distance.shape == (n_points, n_labels, num_houses_per_class, num_windows_per_house)
        outer_window_distance = (1 - window_containment) * window_distance

        return dict(
            embeddings=points,
            crisp_frame_containment=frame_containment,
            crisp_window_containment=window_containment,
            crisp_house_containment=house_containment,
            crisp_containment=containment,
            inner_frame_distance=inner_frame_distance,
            outer_frame_distance=outer_frame_distance,
            inner_window_distances=inner_window_distance,
            outer_window_distances=outer_window_distance,
        )

    def _inside(self, directions, points):
        # frame_containment:  X.shape == (n_labels, num_houses_per_class, n_dim, n_points)
        # window_containment: X.shape == (n_labels, num_houses_per_class, num_windows_per_house, n_dim, n_points)
        X = torch.linalg.solve(directions.transpose(-2, -1), points.T)
        X = X.movedim(-1, 0)
        return torch.all(X >= 0, dim=-1).to(torch.int)

    def _distance(self, directions, points, p=2):
        # TODO: margin
        # line = a + n*t
        # dist(p, a + n*t) = || (a-p) - dot((a-p), n) * n ||
        # with a = (0, ..., 0)
        #      dist(p, n*t) = || -p - dot(-p, n) * n ||
        norms = torch.linalg.vector_norm(directions, ord=p, dim=-1).unsqueeze(-1)
        unit_directions = directions / norms
        dot_products = (-points @ unit_directions.transpose(-2, -1)).unsqueeze(-1)
        dot_times_unit = dot_products * unit_directions.unsqueeze(-3)
        almost = -points.unsqueeze(1) - dot_times_unit
        result = torch.linalg.vector_norm(almost, ord=p, dim=-1)
        return torch.min(result, dim=-1)[0].movedim(-1, 0)

    def _tensor_to_patch(self, tensor):
        direction1 = tensor[0]
        unit_direction1 = direction1 / torch.linalg.vector_norm(direction1)
        direction2 = tensor[1]
        unit_direction2 = direction2 / torch.linalg.vector_norm(direction2)
        LARGE_DISTANCE = 1000.0
        coords = ((0., 0.),
                  (unit_direction1[0].item() * LARGE_DISTANCE, unit_direction1[1].item() * LARGE_DISTANCE), 
                  (unit_direction2[0].item() * LARGE_DISTANCE, unit_direction2[1].item() * LARGE_DISTANCE), 
                  (0., 0.))
        return shapely.Polygon(coords)

    def get_patches(self) -> Iterable[shapely.Geometry]:
        towns = self.town_windows_tensor.shape[0]
        houses = self.town_windows_tensor.shape[1]
        windows = self.town_windows_tensor.shape[2]
        if towns == 1 and houses == 1 and windows == 1:
            return [self._tensor_to_patch(self.town_frame_tensor[0, 0]), self._tensor_to_patch(self.town_windows_tensor[0, 0, 0])]
        town_shapes = []
        for town in range(towns):
            house_shapes = []
            for house in range(houses):
                house_shape = self._tensor_to_patch(self.town_frame_tensor[town, house])
                for window in range(windows):
                    window_shape = self._tensor_to_patch(self.town_windows_tensor[town, house, window])
                    house_shape = shapely.difference(house_shape, window_shape)
                house_shapes.append(house_shape)
            town_shapes.append(shapely.union_all(house_shapes))
        return town_shapes

    def render(self, ax, colors):
        polygons = []
        for shape, color in zip(self.get_patches(), colors):
            if isinstance(shape, shapely.MultiPolygon):
                for geom in shape.geoms:
                    xs, ys = geom.exterior.xy
                    polygons += ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)
            else:
                xs, ys = shape.exterior.xy
                polygons += ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)
        return polygons

class ConeTownModel(torch.nn.Module):
    def __init__(self, embedding_model, out_dim: int, num_houses_per_class=5, num_windows_per_house=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        self.embedding_model = embedding_model
        self.box_model = ConeGeometryNet(embedding_dimensions=self.embedding_model.out_dim, num_classes=out_dim, num_houses_per_class=num_houses_per_class, num_windows_per_house=num_windows_per_house)

    def forward(self, x, **kwargs):
        embeddings = self.embedding_model(x, **kwargs)
        distances = self.box_model(embeddings)
        return distances