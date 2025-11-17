import torch
import torch.nn as nn
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import sys
import os

class ClassificationHeadDummy(nn.Module):
    """This is an example classification head. Don't use in production!"""

    def __init__(self, embedding_dim=512, num_classes=10, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, boxes):
        return self.classifier(boxes)


class YOLOWithClassifier(nn.Module):
    """Combined YOLO11 detection + classification model"""

    def __init__(self, yolo_model_path='yolo11s.pt', num_classes=10,
                 embedding_dim=512, freeze_yolo=False):
        super().__init__()
        # Load YOLO11 model
        self.yolo = YOLO(yolo_model_path, task="detect")
        self.yolo_model = self.yolo.model

        # Optionally freeze YOLO weights, if we do not want to co-train cones and YOLO
        if freeze_yolo:
            for param in self.yolo_model.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = ClassificationHeadDummy(embedding_dim, num_classes)

    def extract_embeddings(self, features, detections):
        boxes = []

        for i, det in enumerate(detections):
            batch_boxes = []
            if det.boxes is not None and len(det.boxes) > 0:
                for box in det.boxes.xyxy:
                    # Crop and pool features from detection region
                    x1, y1, x2, y2 = box.int()
                    # Extract region
                    region = features[i, :, y1:y2, x1:x2]
                    batch_boxes.append((box,region))
            boxes.append(batch_boxes)

        return boxes

    def forward(self, x):
        """
        Forward pass through YOLO and classifier
        Returns: detections, classifications (if extract_features=True)
        """

        with torch.set_grad_enabled(not self.training):
            # Detect boxes
            # Settign IOU here is important, because we have overlapping boxes!
            # See: https://www.ultralytics.com/glossary/intersection-over-union-iou
            detections = self.yolo(x, conf=0, iou=0.95)

        # Extract embeddings for detected objects
        boxes = self.extract_embeddings(x, detections)

        if boxes is not None:
            # Classify detected objects
            classifications = self.classifier(boxes)
            return detections, classifications, boxes

        return detections, None, None


class YOLOClassificationDataset(Dataset):
    """Dataset for YOLO detection + classification"""

    def __init__(self, images_dir, labels_dir, img_size=640):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.image_files = list(self.images_dir.glob('*.jpg')) + \
                           list(self.images_dir.glob('*.png'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Load labels (YOLO format: class x_center y_center width height)
        label_path = self.labels_dir / (img_path.stem + '.txt')
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    labels.append(list(map(float, line.strip().split())))

        return img, torch.tensor(labels) if labels else torch.zeros((0, 5))


def collate_fn(batch):
  return torch.stack([x[0] for x in batch]), [x[1] for x in batch]


def train_epoch(model, dataloader, optimizer, detection_criterion,
                classification_criterion, device):

    total_loss = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        #targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        detections, classifications, boxes = model(images)

        # Compute detection loss (YOLO's built-in loss)
        det_loss = detection_criterion(detections, targets)

        # Compute classification loss if we have detections
        cls_loss = 0
        if classifications is not None and len(classifications) > 0:
            # Assuming targets contains class labels
            # You'll need to match detections to ground truth
            cls_targets = targets[:, 0].long()  # Extract class labels
            if len(cls_targets) == len(classifications):
                cls_loss = classification_criterion(classifications, cls_targets)

        # Combined loss
        loss = det_loss + cls_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main(data_path):
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classification_classes = 10
    batch_size = 8
    learning_rate = 0.001
    epochs = 50

    # Initialize model
    model = YOLOWithClassifier(
        yolo_model_path='yolo11s.pt',
        num_classes=num_classification_classes,
        embedding_dim=512,
        freeze_yolo=False  # Set to True to only train classifier
    ).to(device)

    # Dataset and dataloader
    train_dataset = YOLOClassificationDataset(
        images_dir=os.path.join(data_path,'train/images'),
        labels_dir=os.path.join(data_path, 'train/labels'),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, collate_fn=collate_fn)


    detection_criterion = nn.MSELoss()  # TODO: Figure out how to integrate YOLO-loss
    classification_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam([
        {'params': model.yolo_model.parameters(), 'lr': learning_rate},
        {'params': model.classifier.parameters(), 'lr': learning_rate}
    ])

    # Training loop
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer,
                           detection_criterion, classification_criterion,
                           device)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'checkpoint_epoch_{epoch + 1}.pt')

    print("Training complete!")


if __name__ == '__main__':
    main(sys.argv[1])