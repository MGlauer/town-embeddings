import torch
import typing
import shapely

Point = torch.TensorType


class FrameShape:

    @classmethod
    def tensor_shape(cls) -> typing.Tuple[int]:
        raise NotImplementedError

    def distance_to_points(self, point: Point, p=2):
        raise NotImplementedError

    def contains_points(self, point):
        raise NotImplementedError

    def contains_shape(self, other):
        raise NotImplementedError

    def _from_tensor(self):
        raise NotImplementedError

    @classmethod
    def batch_contains_and_distance_points(cls, shapes: typing.Iterable["Box"], points: torch.Tensor):
        raise NotImplementedError

    def intersection(self, other):
        raise NotImplementedError

    def get_patch(self) -> shapely.Geometry:
        raise NotImplementedError


class Box(FrameShape):

    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.corner1 = tensor[0]
        self.corner2 = tensor[1]

    def contains_points(self, item: Point):
        min_corner = torch.minimum(self.corner1, self.corner2)
        max_corner = torch.maximum(self.corner1, self.corner2)
        return torch.prod((min_corner <= item) * (item <= max_corner), dim=-1)

    def _norm(self, point, p=2, dim=-1):
        if p is None:
            return torch.max(torch.abs(point), dim=dim)[0]
        else:
            return torch.sum(torch.abs(point)**p, dim=dim)**(1/p)

    def distance_to_points(self, point: Point, p=2):
        min_corner = torch.minimum(self.corner1, self.corner2)
        max_corner = torch.maximum(self.corner1, self.corner2)
        margin =  0.1 * (max_corner - min_corner).detach()
        inside = torch.prod((min_corner + margin <= point) * (point <= max_corner - margin), dim=-1)
        outside = 1-torch.prod((min_corner - margin <= point) * (point <= max_corner + margin), dim=-1
                               )
        outer_distance = (1-inside) * self._norm(torch.relu(min_corner + margin - point) + torch.relu(point - max_corner + margin), dim=-1, p=p)
        inner_distance = (1-outside) * self._norm(torch.minimum(torch.relu(max_corner + margin - point), torch.relu(point - min_corner + margin)), dim=-1, p=p)

        center_points = (min_corner + max_corner)/2

        center_distances = (1-outside) * self._norm(center_points - point, dim=-1, p=p)

        return dict(outside=outer_distance, inside=inner_distance, center=center_distances)

    @classmethod
    def tensor_shape(cls):
        return (2,)

    def get_patch(self) -> shapely.Geometry:
        min_corner = torch.minimum(self.corner1, self.corner2)
        max_corner = torch.maximum(self.corner1, self.corner2)
        return shapely.box(min_corner[0].item(), min_corner[1].item(), max_corner[0].item(), max_corner[1].item())

    def intersection(self, other):
        if other == NOTHING:
            return NOTHING
        s_min = torch.minimum(self.corner1, self.corner2)
        s_max = torch.maximum(self.corner1, self.corner2)
        o_min = torch.minimum(other.corner1, other.corner2)
        o_max = torch.maximum(other.corner1, other.corner2)
        d_min = torch.minimum(s_max, o_max)
        d_max = torch.maximum(s_min, o_min)
        if torch.prod(d_max <= d_min):
            return Box(torch.clone(torch.stack((d_min, d_max), dim=0)).detach())
        else:
            return NOTHING

    def get_size(self):
        return torch.sum(torch.abs(self.corner1 - self.corner2),dim=0)


class _All(FrameShape):

    def intersection(self, other):
        return other

    def contains_points(self, points: torch.Tensor):
        return torch.ones((points.shape[0],), device=points.device)

    def contains_shape(self, other):
        return True


class _Nothing(FrameShape):

    def intersection(self, other):
        return self

    def contains_points(self, points: Point):
        return torch.zeros((points.shape[0],), device=points.device)

    def contains_shape(self, other):
        return False

    def get_patch(self) -> shapely.Geometry:
        return None


NOTHING = _Nothing()
ALL = _All()

if __name__=="__main__":
    from matplotlib import pyplot as plt
    import seaborn as sns
    box = Box(torch.tensor(((10, 20), (20, 40))))
    ranges = torch.linspace(0, 50, 500)
    a, b = torch.meshgrid(ranges, ranges, indexing="xy")
    distance = box.distance_to_points(torch.stack((a,torch.flip(b, dims=(0,))), dim=-1))
    ax = plt.gca()
    shape = box.get_patch()
    plt.imshow(distance, cmap='hot', extent=(0, 50, 0, 50))
    xs, ys = shape.exterior.xy
    ax.fill(xs, ys, fc=(0,1,0), ec='k', alpha=0.2)
    plt.show()