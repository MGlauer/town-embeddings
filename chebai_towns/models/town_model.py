from typing import Dict, Any, Iterable

from chebai.preprocessing.structures import XYData
from chebai_towns.towns.shape import FrameShape, Box
from chebai_towns.towns.geometry import House, Town
import torch
from torch.nn import Module
from chebai.models import ChebaiBaseNet
from chebai_towns.towns.shape import Box
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

        new = dict(
            embeddings=points,
            frame_containment=frame_containment,
            window_containment=window_containment,
            house_containment=house_containment,
            containment=containment,
            inner_frame_distance=inner_frame_distance,
            outer_frame_distance=outer_frame_distance,
            inner_window_distances=inner_window_distance,
            outer_window_distances=outer_window_distance,
        )

        return new

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
        for shape, color in zip(self.get_patches(), colors):
            if isinstance(shape, shapely.MultiPolygon):
                for geom in shape.geoms:
                    xs, ys = geom.exterior.xy
                    ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)
            else:
                xs, ys = shape.exterior.xy
                ax.fill(xs, ys, fc=color, ec='k', alpha=0.1)

class TownModel(ChebaiBaseNet):
    def __init__(self, embedding_model: ChebaiBaseNet, out_dim: int, num_houses_per_class=5, num_windows_per_house=4, *args, **kwargs):
        super().__init__(*args, **kwargs, exclude_hyperparameter_logging=["embedding_model"])
        self.automatic_optimization = False
        self.embedding_model = embedding_model
        self.box_model = GeometryNet(embedding_dimensions=self.embedding_model.out_dim, num_classes=out_dim, num_houses_per_class=num_houses_per_class, num_windows_per_house=num_windows_per_house)

    def forward(self, x, **kwargs):
        embedding_output = self.embedding_model(x, **kwargs)

        embeddings = embedding_output.pop("logits")

        #for t in self.box_model.towns:
        #    t.consolidate()
        distances = self.box_model(embeddings)

        return dict(output=distances, embeddings=embeddings, **embedding_output)

    def _process_batch(self, batch: XYData, batch_idx: int) -> Dict[str, Any]:
        return self.embedding_model._process_batch(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        for op in self.optimizers():
            op.zero_grad()

        output = super().training_step(batch, batch_idx)

        output["loss"].backward()

        for op in self.optimizers():
            op.step()

        return output

    def _get_prediction_and_labels(
        self, data: Dict[str, Any], labels: torch.Tensor, output: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        return output["output"]["containment"], labels

    def configure_optimizers(self):
        emb_optimizer = torch.optim.Adam(self.embedding_model.parameters(), lr=2e-3)
        model_optimizer = torch.optim.Adam(self.box_model.parameters(), lr=1e-3)
        return emb_optimizer, model_optimizer


