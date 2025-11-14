from typing import Any
from unicodedata import name
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.engine.results import Results
from ultralytics.utils import RANK
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8DetectionLoss,
)
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from potemkin.io.render import draw_2d_on_axes
from potemkin.loss.town_loss import DistanceLoss
from potemkin.models.town_model import ConeTownModel
from matplotlib import colormaps


class ConeLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.model = model
        self.features = None
        # The layer from which to extract the features / embeddings from
        #feature_layer: tuple[str, torch.nn.Module] = list(self.model.model.named_modules())[-3]
        feature_layer: torch.nn.Module = dict(self.model.model.named_modules())["23.cv2.2"]
        #for name, m in model.model.named_modules():
        #    print(name)
        feature_layer.register_forward_hook(self.feature_layer_hook)
        self.loss = E2EDetectLoss(model) if getattr(model, "end2end", False) else v8DetectionLoss(model)

    def feature_layer_hook(self, module, input, output):
        self.features = output

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        print("----------------------------------\n")
        print("Features shape:", self.features.shape if self.features is not None else "None")
        print("Preds keys:", )
        print("----------------------------------\n")
        return self.loss(preds, batch)


class ConeModel(DetectionModel):

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return ConeLoss(self)


class ConeTrainer(DetectionTrainer):

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = ConeModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model


model = YOLO("yolo11s.pt")
model.train(
    trainer=ConeTrainer,
    data="pizzaiolo/DEMO_SAMPLE_pizzaiolo_dataset_YOLO/data.yaml", 
    epochs=10
)
results: list[Results] = model("pizzaiolo/DEMO_SAMPLE_pizzaiolo_dataset_YOLO/test/images")

n_dim = 2
n_points = len(results)
n_labels = len(results[0].names)

points = torch.tensor(range(n_points), dtype=torch.int)
labels = torch.zeros((n_points, n_labels), dtype=torch.bool)
result: Results
for i, result in enumerate(results):
    assert result.boxes is not None, "result.boxes should not be None after detection task"
    for class_id in result.boxes.cls:
        labels[i, int(class_id)] = 1

print("-------------------------------")
print(labels)

class SimpleEmbedding(torch.nn.Module):
    def __init__(self, n_points, n_dim):
        super(SimpleEmbedding, self).__init__()
        self.out_dim = n_dim
        self.points = torch.nn.Parameter(torch.rand((n_points, n_dim))*6-3)

    def forward(self, idx):
        return self.points[idx]

embedding_model = SimpleEmbedding(n_points, n_dim)
embeddings = embedding_model(points)

model = ConeTownModel(embedding_model, n_labels, num_houses_per_class=1, num_windows_per_house=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
cmap = colormaps["Set1"]

colors = list(cmap.colors)[:n_labels]

loss_fn = DistanceLoss()

fig, ax = plt.subplots()
ax.set_xlim((-5, 5))
ax.set_ylim((-5, 5))
figs = []

for _ in range(250):
    optimizer.zero_grad()
    embeddings, distances = model(points)
    loss = loss_fn(distances, labels)
    accuracy = (distances['crisp_containment']==labels).float().mean(dim=-1)
    point_colors = torch.stack((1-accuracy, accuracy, 0*accuracy)).T.tolist()

    figs.append(draw_2d_on_axes(embeddings, ax, model, point_colors=point_colors, town_colors=colors))

    loss.backward()
    optimizer.step()

ani = animation.ArtistAnimation(fig, figs, interval=100)
ani.save("animation.gif", writer="imagemagick", fps=10)