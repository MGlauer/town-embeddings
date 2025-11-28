import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import nms, box_iou
from ultralytics import YOLO

from ultralytics.utils.loss import v8DetectionLoss
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import sys
import os

from potemkin.loss.town_loss import DistanceLoss
from potemkin.models.town_model import ConeGeometryNet


class YOLOWithClassifier(nn.Module):
    """Combined YOLO11 detection + classification model"""

    def __init__(self, yolo_model_path='yolo11s.pt', num_classes=10,
                 embedding_dim=3, freeze_yolo=False, num_houses_per_class=5, num_windows_per_house=4,conf_thresh=0.1,iou_threshold=0.5):
        super().__init__()
        # Load YOLO11 model
        self.yolo = YOLO(yolo_model_path, task="detect")
        self.yolo_model = self.yolo.model
        no = tuple(self.yolo_model.model)[-1].no
        self.embedding_map = torch.nn.Linear(no, embedding_dim)
        self.cone_model = ConeGeometryNet(embedding_dimensions=embedding_dim, num_classes=num_classes,
                                          num_houses_per_class=num_houses_per_class,
                                          num_windows_per_house=num_windows_per_house)

        # Optionally freeze YOLO weights, if we do not want to co-train cones and YOLO
        if freeze_yolo:
            for param in self.yolo_model.parameters():
                param.requires_grad = False

        # Classification head#
        self.embedding_dim = embedding_dim
        self.conf_thresh = conf_thresh
        self.iou_threshold = iou_threshold

    def extract_boxes_and_embeddings(self, raw_boxes, feature_maps, img_size):
        """
        raw_boxes: Tensor (B, N, 4) with last dimension (xc, yc, w, h)
        feature_maps: list of 3 embedding tensors [(B,C,H1,W1), (B,C,H2,W2), (B,C,H3,W3)]
        img_size: tuple (H_img, W_img)

        Returns:
            boxes_xyxy: Tensor (B, N, 4) in pixel coordinates
            embeddings: Tensor (B, N, C) embeddings per box
        """
        num_batches, num_boxes, _ = raw_boxes.shape
        h_img, w_img = img_size
        c = feature_maps[0].shape[1]  # channels

        xc = raw_boxes[..., 0]
        yc = raw_boxes[..., 1]
        w = raw_boxes[..., 2]
        h = raw_boxes[..., 3]


        # Convert to xyxy
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        # Embeddings should have dimension
        embeddings = torch.zeros(num_batches, num_boxes, c, device=raw_boxes.device)

        # Decide which scale to use based on box area
        box_area = w * h
        img_area = w_img * h_img

        # I think these are the correct thresholds. We should double-check later.
        small_thr = 20/640 * img_area
        medium_thr = 40/640 * img_area

        # Class-agnostic NMS
        # Track those boxes that are not too similar to others
        keeps = [ nms(boxes_xyxy[j], raw_boxes[j, :, 4:].max(1)[0], iou_threshold=self.iou_threshold)  for j in range(num_batches)]

        #Todo: I think this can be optimized by first partitioning the boxes by resolution and then processing the results in bulk.
        #      But for now this slow implementation suffices, I guess.
        for j in range(num_batches):
            for i in range(num_boxes):
                # pick scale
                area = box_area[j, i]
                if area <= small_thr:
                    fmap = feature_maps[0][j]  # (C,H,W)
                elif area <= medium_thr:
                    fmap = feature_maps[1][j]
                else:
                    fmap = feature_maps[2][j]

                _, H_f, W_f = fmap.shape

                # map box to feature map coordinates
                fx1 = x1[j, i] / w_img * W_f
                fy1 = y1[j, i] / h_img * H_f
                fx2 = x2[j, i] / w_img * W_f
                fy2 = y2[j, i] / h_img * H_f

                # clamp to feature map
                fx1 = fx1.clamp(0, W_f - 1)
                fy1 = fy1.clamp(0, H_f - 1)
                fx2 = fx2.clamp(0, W_f - 1)
                fy2 = fy2.clamp(0, H_f - 1)

                # crop and average-pool to get embedding
                # Maybe we can use some smarter pooling here.
                embeddings[j, i] = torch.nn.functional.interpolate(
                    fmap[None, :, int(fy1):int(fy2) + 1, int(fx1):int(fx2) + 1],
                    size=(1, 1),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(-1).squeeze(-1)  # shape (C,)
        return boxes_xyxy, embeddings, keeps

    def forward(self, x):
        """
        Forward pass through YOLO and cone model
        Returns: dict
        """
        num_batches = x.shape[0]
        with torch.set_grad_enabled(not self.training):
            raw_preds, feats = self.yolo.model(x)
            #This is a filter that decides whether a box should (not) be dropped.
            boxes_xyxy, embeddings, fltr = self.extract_boxes_and_embeddings(raw_preds.permute(0,2,1), feats, x.shape[2:])

        batch_indices = torch.arange(num_batches, device=embeddings.device).unsqueeze(1).repeat(1,embeddings.shape[1])
        stacked_embeddings = torch.cat([x[f] for x,f in zip(embeddings, fltr)], dim=0)
        stack_batches = torch.cat([x[f] for x, f in zip(batch_indices, fltr)], dim=0)

        # Embeddings still have dimension 144, which is the internal embedding size of YOLO. I suspect that we cannot change this
        # if we want to use a pre-trained model. Therefore, we should add some dimension transformation post-hoc
        transformed_embeddings = self.embedding_map(stacked_embeddings)

        # Classify detected objects using geometric model
        raw_distances = self.cone_model(transformed_embeddings)

        #Todo: Here we untangle distances, I think this is keeps entries in order, but we should double-check!
        for i in range(num_batches):
            d = dict(boxes=boxes_xyxy[i,fltr[i]])
            for k in raw_distances:
                d[k] = raw_distances[k][stack_batches==i]
            yield d
        #return {"embeddings": embeddings, "distances": distances, "boxes": boxes_xyxy, "raw_boxes": raw_preds}



class BBoxLoss(torch.nn.Module):

    def forward(self, pred_boxes, target_boxes):

        ious = box_iou(pred_boxes, target_boxes)
        best_value, best_idx = torch.max(ious, dim=1)
        boxes_with_no_match = (best_value==0)
        # For boxes that do not overlap with any box, we assign a random "best-fitting" box.
        # Todo: Maybe we could choose the closest box here instead.
        best_idx[boxes_with_no_match] = torch.randint_like(best_idx, target_boxes.shape[0])[boxes_with_no_match]
        return torch.sum(1-best_value), best_idx

class YOLOClassificationDataset(Dataset):
    """Dataset for YOLO detection + classification"""

    def __init__(self, images_dir, labels_dir, num_classes, img_size=640):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.image_files = list(self.images_dir.glob('*.jpg')) + \
                           list(self.images_dir.glob('*.png'))
        self.num_classes = num_classes

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
        target_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    target = list(map(float, line.strip().split()))
                    labels.append(target[0])
                    target_boxes.append(target[1:])
        labels = torch.tensor(labels).int()
        onehot_labels = to_onehot(labels, self.num_classes)
        return img, dict(labels=onehot_labels, boxes=torch.tensor(target_boxes))

def to_onehot(x, num_classes):
    # Create one-hot matrix
    M = torch.zeros(num_classes, x.shape[0], dtype=torch.bool)
    M[x, torch.arange(x.shape[0])] = True
    return M

def collate_fn(batch):
  return (torch.stack([x[0] for x in batch]),
          dict(
              labels=[x[1]["labels"] for x in batch],
              boxes=[x[1]["boxes"] for x in batch]))


def train_epoch(model, dataloader, optimizer, detection_criterion,
                distance_loss, device):

    total_loss = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        optimizer.zero_grad()

        loss = 0

        # Forward pass
        for batch_part, result in enumerate(model(images)):


            # Compute detection loss (YOLO's built-in loss)
            target_boxes = targets["boxes"][batch_part].to(device)
            target_label = targets["labels"][batch_part].to(device)
            det_loss, box_assignment = detection_criterion(result["boxes"], target_boxes)


            # Todo: I do not like this part. It may be, that some labels are never assigned early in training.
            #       In this case, the respective label does not contribute to the loss, which is bad
            target = target_label[:,box_assignment].T
            cls_loss = distance_loss(result, target)

            # Combined loss
            loss += det_loss + cls_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main(data_path):
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    learning_rate = 0.001
    epochs = 50

    # TODO: We need to load this dynamically
    num_classification_classes = 24

    # Initialize model
    model = YOLOWithClassifier(
        yolo_model_path='yolo11s.pt',
        num_classes=num_classification_classes,
        embedding_dim=3,
        freeze_yolo=False  # Set to True to only train classifier
    ).to(device)

    # Dataset and dataloader
    train_dataset = YOLOClassificationDataset(
        images_dir=os.path.join(data_path,'train/images'),
        labels_dir=os.path.join(data_path, 'train/labels'),
        num_classes=num_classification_classes,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, collate_fn=collate_fn)


    detection_criterion = BBoxLoss()  # TODO: Figure out how to integrate YOLO-loss
    distance_loss = DistanceLoss()

    optimizer = torch.optim.Adam([
        {'params': model.yolo_model.parameters(), 'lr': learning_rate},
        {'params': model.cone_model.parameters(), 'lr': learning_rate}
    ])

    # Training loop
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer,
                           detection_criterion, distance_loss,
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