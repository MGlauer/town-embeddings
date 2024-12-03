from typing import Dict, Any

from chebai.preprocessing.structures import XYData
from chebai_towns.towns.shape import FrameShape, Box
from chebai_towns.towns.geometry import House, Town
import torch
from torch.nn import Module
from chebai.models import ChebaiBaseNet
from chebai_towns.towns.shape import Box

class GeometryNet(torch.nn.Module):

    def __init__(self, frame_shape_class: type[FrameShape], embedding_dimensions: int, num_classes, num_houses_per_class=5, num_windows_per_house=5,
                 shape_kwargs=None):
        super().__init__()

        tensor_shape = frame_shape_class.tensor_shape()

        self.shape_tensor = torch.nn.Parameter(3-6*torch.rand((num_classes, num_houses_per_class, num_windows_per_house+1, *tensor_shape, embedding_dimensions)))

        self.towns = [
            Town([House(frame_shape_class(self.shape_tensor[c,h,-1]), [frame_shape_class(self.shape_tensor[c,h,w]) for w in range(num_windows_per_house)])
             for h in range(num_houses_per_class)]) for c in range(num_classes)]

        #self.town_frame_tensor = torch.nn.Parameter(3-6*torch.rand((num_classes, num_houses_per_class, *tensor_shape, embedding_dimensions)))
        #self.town_windows_tensor = torch.nn.Parameter(3-6*torch.rand((num_classes, num_houses_per_class, num_windows_per_house+1, *tensor_shape, embedding_dimensions)))

    def forward(self, data):

        return dict(embeddings=data,
            containment=[t.contains_points(data) for t in self.towns],
            distances=[t.detailed_distances(data) for t in self.towns])


class TownModel(ChebaiBaseNet):
    def __init__(self, embedding_model: ChebaiBaseNet, out_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs, exclude_hyperparameter_logging=["embedding_model"])
        self.automatic_optimization = False
        self.embedding_model = embedding_model
        self.box_model = GeometryNet(frame_shape_class=Box, embedding_dimensions=self.embedding_model.out_dim, num_classes=out_dim, num_houses_per_class=3, num_windows_per_house=2)

    def forward(self, x, **kwargs):
        embedding_output = self.embedding_model(x, **kwargs)

        embeddings = embedding_output.pop("logits")

        for t in self.box_model.towns:
            t.consolidate()
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
        return torch.stack(output["output"]["containment"], dim=-1), labels

    def configure_optimizers(self):
        emb_optimizer = torch.optim.Adam(self.embedding_model.parameters(), lr=2e-3)
        model_optimizer = torch.optim.Adam(self.box_model.parameters(), lr=1e-3)
        return emb_optimizer, model_optimizer