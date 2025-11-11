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

        self.town_frame_tensor = torch.nn.Parameter(3-6*torch.rand((num_classes, num_houses_per_class, embedding_dimensions, embedding_dimensions)))
        self.town_windows_tensor = torch.nn.Parameter(3-6*torch.rand((num_classes, num_houses_per_class, num_windows_per_house, embedding_dimensions, embedding_dimensions)))
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

        # inner_frame_distance.shape == (n_points, n_labels, num_houses_per_class)
        inner_frame_distance = frame_containment * self._distance(- self.town_frame_tensor, points, p=self.p)
        # outer_frame_distance.shape == (n_points, n_labels, num_houses_per_class)
        outer_frame_distance = (1 - frame_containment) * self._distance(self.town_frame_tensor, points, p=self.p)

        # inner_window_distance.shape == (n_points, n_labels, num_houses_per_class, num_windows_per_house)
        inner_window_distance = window_containment * self._distance(- self.town_windows_tensor, points, p=self.p)
        # outer_window_distance.shape == (n_points, n_labels, num_houses_per_class, num_windows_per_house)
        outer_window_distance = (1 - window_containment) * self._distance(self.town_windows_tensor, points, p=self.p)

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
        A = directions.unsqueeze(-3)
        x = points.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        b = A @ x
        b = b.squeeze(-1)
        b = b.movedim(-2, 0)
        return torch.all(b <= 0, dim=-1).to(torch.int)

    def _distance(self, directions, points, p=2):
        A = directions.unsqueeze(-3)
        x = points.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        b = A @ x
        b = b.squeeze(-1)
        b = b.movedim(-2, 0)
        return torch.linalg.vector_norm(torch.relu(b), ord=p, dim=-1)

    def _find_direction_bound(self, direction, xlim, ylim):
        assert direction.shape == (2,), "Can only create patches from single 2D direction vectors"
        if direction[0] < 0:
            x0 = xlim[0]
            d0 = 2
        else:
            x0 = xlim[1]
            d0 = 0
        y0 = direction[1] / direction[0] * x0
        if direction[1] < 0:
            y1 = ylim[0]
            d1 = 3
        else:
            y1 = ylim[1]
            d1 = 1
        x1 = direction[0] / direction[1] * y1
        if abs(y0) < abs(y1):
            return x0, y0, d0
        else:
            return x1, y1, d1
        
    def _get_half_plane_polygon(self, direction1: torch.Tensor, direction2: torch.Tensor, xlim, ylim) -> shapely.Polygon:
        """
        The half-plane bordered by direction1 and direction2 should be left of direction1 and right of direction2.
        """
        patch_corners = [(0., 0.)]
        x0, y0, d0 = self._find_direction_bound(direction1, xlim, ylim)
        patch_corners.append((x0, y0))
        x1, y1, d1 = self._find_direction_bound(direction2, xlim, ylim)
        if d0 != d1:
            quadrant_corners = {0: (xlim[1], ylim[1]),
                                1: (xlim[0], ylim[1]),
                                2: (xlim[0], ylim[0]),
                                3: (xlim[1], ylim[0])}
            while d0 != d1:
                corner = quadrant_corners[d0]
                patch_corners.append(corner)
                d0 = (d0 + 1) % 4
        patch_corners.append((x1, y1))
        patch_corners.append((0., 0.))
        return shapely.Polygon(patch_corners)

    def _tensor_to_patch(self, tensor: torch.Tensor, xlim, ylim) -> shapely.Geometry:
        assert tensor.shape == (2, 2), "Can only create patches from 2d cone with 2 constraints"
        assert xlim is not None and ylim is not None, "Need xlim and ylim to create patches"
        # The cone is defined through:
        #     ┏           ┓     ┏   ┓      ┏   ┓
        #     ┃ c1_1 c1_2 ┃  *  ┃ x ┃  <=  ┃ 0 ┃
        #     ┃ c2_1 c2_2 ┃     ┃ y ┃      ┃ 0 ┃
        #     ┗           ┛     ┗   ┛      ┗   ┛
        # 
        # <=> c1_1 * x + c1_2 * y <= 0
        #     c2_1 * x + c2_2 * y <= 0
        constraint1 = tensor[0].detach()
        constraint2 = tensor[1].detach()
        # Vector (c_1, c_2) is orthogonal to line c_1 * x + c_2 * y = 0 and points towards the side where c_1 * x + c_2 * y > 0
        # Define direction vectors so that contraint vector is to its right side, since _get_half_plane_polygon() expects half-plane (<= 0) to be left of the direction vector
        direction1 = torch.tensor([- constraint1[1], constraint1[0]])
        direction2 = torch.tensor([- constraint2[1], constraint2[0]])
        polygon1 = self._get_half_plane_polygon(direction1, - direction1, xlim, ylim)
        polygon2 = self._get_half_plane_polygon(direction2, - direction2, xlim, ylim)
        return shapely.intersection(polygon1, polygon2)

    def get_patches(self, xlim, ylim) -> Iterable[shapely.Geometry]:
        towns = self.town_windows_tensor.shape[0]
        houses = self.town_windows_tensor.shape[1]
        windows = self.town_windows_tensor.shape[2]
        if towns == 1 and houses == 1 and windows == 1:
            return [self._tensor_to_patch(self.town_frame_tensor[0, 0], xlim, ylim), self._tensor_to_patch(self.town_windows_tensor[0, 0, 0], xlim, ylim)]
        town_shapes = []
        for town in range(towns):
            house_shapes = []
            for house in range(houses):
                house_shape = self._tensor_to_patch(self.town_frame_tensor[town, house], xlim, ylim)
                for window in range(windows):
                    window_shape = self._tensor_to_patch(self.town_windows_tensor[town, house, window], xlim, ylim)
                    house_shape = shapely.difference(house_shape, window_shape)
                house_shapes.append(house_shape)
            town_shapes.append(shapely.union_all(house_shapes))
        return town_shapes

    def render(self, ax, colors):
        polygons = []
        shape: shapely.Geometry
        for shape, color in zip(self.get_patches(ax.get_xlim(), ax.get_ylim()), colors):
            # if shape.is_empty:
            #     continue
            if isinstance(shape, shapely.GeometryCollection):
                for geom in shape.geoms:
                    if isinstance(geom, shapely.Polygon):
                        xs, ys = geom.exterior.xy
                        polygons += ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)
            elif isinstance(shape, shapely.MultiPolygon):
                for geom in shape.geoms:
                    xs, ys = geom.exterior.xy
                    polygons += ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)
            elif isinstance(shape, shapely.Polygon):
                 xs, ys = shape.exterior.xy
                 polygons += ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)
            # else:
            #     xs, ys = shape.exterior.xy
            #     polygons += ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)
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
        return embeddings, distances