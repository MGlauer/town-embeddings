import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from potemkin.io.render import draw_2d_on_axes
from potemkin.loss.town_loss import DistanceLoss
from potemkin.models.town_model import ConeTownModel
from matplotlib import colormaps

model = YOLO("yolo11s.pt")
model.train(data="pizzaiolo/DEMO_SAMPLE_pizzaiolo_dataset_YOLO/data.yaml", epochs=10)
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