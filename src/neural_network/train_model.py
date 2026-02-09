import os
import random
import math
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F  # image transforms (hflip, to_tensor)
import torch.nn as nn
import torch.nn.functional as F_nn  # neural network losses, softmax, etc.

# ---------- USER CONFIG ----------
IMAGES_DIR = r"C:\Users\Heisenberg\Downloads\sistem-de-inteligenta-artificiala-cu-scopul-de-a-detecta-sobolani-rozatoare--main\data\processed"
CSV_PATH   = r"C:\Users\Heisenberg\Downloads\sistem-de-inteligenta-artificiala-cu-scopul-de-a-detecta-sobolani-rozatoare--main\data\generated\etichete256.csv"
OUTPUT_MODEL_DIR = r"C:\Users\Heisenberg\Downloads\sistem-de-inteligenta-artificiala-cu-scopul-de-a-detecta-sobolani-rozatoare--main\src\neural_network"
MODEL_PATH = r"./saved_models/untrained_model.pth"
# ---------------------------------

NUM_EPOCHS = 30
BATCH_SIZE = 4
# reduced LR to avoid exploding gradients; adjust if you know what you're doing
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 8
LR_GAMMA = 0.1
VAL_SPLIT = 0.2
RANDOM_SEED = 42
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]
GRAD_CLIP_NORM = 5.0
# ---------------------------------

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Mapare etichete: implicit facem MERGE la o clasa "rat".
def map_label_to_id(label_name):
    # versiune simplă: toate la 1
    return 1

class RatsDataset(Dataset):
    def __init__(self, images_dir, csv_path, transforms=None):
        self.images_dir = Path(images_dir)
        # read csv if exists; if not, create empty df with expected columns
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = pd.DataFrame(columns=['label_name','image_name','bbox_x','bbox_y','bbox_width','bbox_height','image_width','image_height'])
        # list image files present in images_dir
        files = []
        if self.images_dir.exists():
            for p in self.images_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    files.append(p.name)
        # union of files in folder and image names referenced in CSV
        csv_images = []
        if 'image_name' in self.df.columns:
            csv_images = [str(x) for x in self.df['image_name'].unique() if pd.notna(x)]
        all_names = sorted(set(files) | set(csv_images))
        self.image_names = all_names
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.images_dir / img_name
        if not img_path.exists():
            # If the CSV referenced an image that is not present on disk, raise explicit error
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Open image (PIL). If grayscale, convert to RGB (model expects 3 channels).
        img = Image.open(img_path).convert("RGB")

        # Get all rows for this image (may be empty -> negative image)
        if 'image_name' in self.df.columns:
            rows = self.df[self.df['image_name'] == img_name]
        else:
            rows = self.df.iloc[0:0]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for _, r in rows.iterrows():
            x = float(r['bbox_x'])
            y = float(r['bbox_y'])
            w = float(r['bbox_width'])
            h = float(r['bbox_height'])
            # Convert (x,y,w,h) -> (x_min,y_min,x_max,y_max)
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            # clip to image if image dims provided in CSV
            if 'image_width' in r and not pd.isna(r['image_width']):
                im_w = float(r['image_width'])
            else:
                im_w = float(img.width)
            if 'image_height' in r and not pd.isna(r['image_height']):
                im_h = float(r['image_height'])
            else:
                im_h = float(img.height)
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            x_max = min(im_w, x_max)
            y_max = min(im_h, y_max)
            if x_max <= x_min or y_max <= y_min:
                # skip degenerate boxes
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(map_label_to_id(r['label_name']))  # class ids (1..N)
            areas.append((x_max - x_min) * (y_max - y_min))
            iscrowd.append(0)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([idx])

        # include original image width/height so model can normalize targets
        orig_size = torch.tensor([float(img.width), float(img.height)], dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
            "orig_size": orig_size
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        # Convert PIL image to tensor
        img_tensor = F.to_tensor(img)  # converts to [0,1] float tensor (C,H,W)
        return img_tensor, target

# Simple transforms that also adjust boxes
class ComposeTransforms:
    def __init__(self, hflip_prob=0.5, random_crop=False):
        self.hflip_prob = hflip_prob
        self.random_crop = random_crop

    def __call__(self, image, target):
        # Horizontal flip
        if random.random() < self.hflip_prob:
            image = F.hflip(image)
            w, h = image.size
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

# -------------------
# SimpleDetector class (included so we can load a state_dict from disk)
# - predicts normalized box coordinates in [0,1] using sigmoid
# - compares predictions against normalized ground-truth boxes
# -------------------
class SimpleDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # small conv backbone -> fixed 1x1 feature map
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        feat_dim = 64

        # classification head -> logits
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_classes)
        )

        # bbox head -> 4 coordinates (normalized via sigmoid in forward)
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, images, targets=None):
        # accept Tensor[N,C,H,W], Tensor[C,H,W], or list[Tensor]
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            imgs = [img for img in images]
        elif isinstance(images, torch.Tensor) and images.dim() == 3:
            imgs = [images]
        else:
            imgs = list(images)

        device = next(self.parameters()).device
        imgs = [img.to(device) for img in imgs]

        cls_logits_list = []
        box_pred_list = []

        for img in imgs:
            x = img.unsqueeze(0)  # 1,C,H,W
            feats = self.backbone(x)  # 1,feat,1,1
            cls_logits = self.cls_head(feats)  # 1,num_classes
            box_pred = self.box_head(feats)    # 1,4
            cls_logits_list.append(cls_logits)
            box_pred_list.append(box_pred)

        cls_logits = torch.cat(cls_logits_list, dim=0)  # N,num_classes
        box_pred = torch.cat(box_pred_list, dim=0)      # N,4

        # apply sigmoid to box predictions to constrain to (0,1)
        box_pred = torch.sigmoid(box_pred)

        if targets is not None:
            # compute classification and bbox regression losses against normalized GT boxes
            total_cls_loss = torch.tensor(0.0, device=cls_logits.device)
            total_bbox_loss = torch.tensor(0.0, device=box_pred.device)
            count_cls = 0
            count_bbox = 0

            for i, tgt in enumerate(targets):
                if tgt is None:
                    continue

                # classification: use first label if present
                if "labels" in tgt and tgt["labels"].numel() > 0:
                    gt_label = tgt["labels"][0].to(cls_logits.device).long().unsqueeze(0)
                    logits = cls_logits[i].unsqueeze(0)
                    cls_loss = F_nn.cross_entropy(logits, gt_label)
                    total_cls_loss = total_cls_loss + cls_loss
                    count_cls += 1

                # bbox: normalize GT to [0,1] using original image size
                if "boxes" in tgt and tgt["boxes"].numel() > 0:
                    gt_box = tgt["boxes"][0].to(box_pred.device).unsqueeze(0)  # (1,4) pixel coords
                    if "orig_size" in tgt:
                        orig = tgt["orig_size"].to(box_pred.device)  # (2,) -> (w,h)
                        denom = torch.tensor([orig[0], orig[1], orig[0], orig[1]], device=box_pred.device)
                        # avoid division by zero
                        denom = torch.clamp(denom, min=1.0)
                        gt_box_norm = gt_box / denom
                    else:
                        # fallback: assume 256x256 if orig size not present
                        denom = torch.tensor([256.0, 256.0, 256.0, 256.0], device=box_pred.device)
                        gt_box_norm = gt_box / denom

                    pred_box = box_pred[i].unsqueeze(0)  # (1,4) normalized
                    bbox_loss = F_nn.mse_loss(pred_box, gt_box_norm)
                    total_bbox_loss = total_bbox_loss + bbox_loss
                    count_bbox += 1

            if count_cls > 0:
                total_cls_loss = total_cls_loss / count_cls
            else:
                total_cls_loss = torch.tensor(0.0, device=cls_logits.device)

            if count_bbox > 0:
                total_bbox_loss = total_bbox_loss / count_bbox
            else:
                total_bbox_loss = torch.tensor(0.0, device=box_pred.device)

            loss_dict = {
                "loss_classifier": total_cls_loss,
                "loss_box_reg": total_bbox_loss
            }
            return loss_dict

        # inference path -> produce detections in normalized coordinates
        probs = F_nn.softmax(cls_logits, dim=1)
        scores, labels = torch.max(probs, dim=1)

        # ensure box format (x_min,y_min,x_max,y_max) and non-negative values
        x_min = torch.min(box_pred[:, 0], box_pred[:, 2]).unsqueeze(1)
        x_max = torch.max(box_pred[:, 0], box_pred[:, 2]).unsqueeze(1)
        y_min = torch.min(box_pred[:, 1], box_pred[:, 3]).unsqueeze(1)
        y_max = torch.max(box_pred[:, 1], box_pred[:, 3]).unsqueeze(1)
        boxes = torch.cat([x_min, y_min, x_max, y_max], dim=1)
        boxes = torch.clamp(boxes, min=0.0, max=1.0)

        detections = []
        for i in range(len(imgs)):
            detections.append({
                "boxes": boxes[i].unsqueeze(0),
                "labels": labels[i].unsqueeze(0),
                "scores": scores[i].unsqueeze(0)
            })
        return detections

# -------------------
# get_model: try loading from MODEL_PATH (state_dict), otherwise fall back to torchvision's Faster R-CNN
# -------------------
def get_model(num_classes):
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        print("Loading SimpleDetector state_dict from:", MODEL_PATH)
        model = SimpleDetector(num_classes=num_classes)  # architecture must match saved state
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        return model

    # fallback: build a Faster R-CNN (no pretrained weights)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        # ensure tensors in targets are on device
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        # if model returned detections instead of losses, skip this batch
        if isinstance(loss_dict, list):
            print("Warning: model returned detections during training batch; skipping batch.")
            continue

        losses = sum(loss for loss in loss_dict.values())

        # check for finite loss
        if not torch.isfinite(losses):
            print(f"Non-finite loss encountered at epoch {epoch} iter {i}: {losses.item()}")
            continue

        optimizer.zero_grad()
        losses.backward()

        # gradient clipping to avoid explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()

        running_loss += losses.item()
        if i % print_freq == 0:
            print(f"Epoch {epoch} Iter {i}/{len(data_loader)} Loss: {losses.item():.6f}")
    avg_loss = running_loss / max(1, len(data_loader))
    print(f"Epoch {epoch} average loss: {avg_loss:.6f}")
    return avg_loss

def evaluate_on_loader(model, data_loader, device):
    """
    Calculează loss-ul mediu pe loader-ul de validare.
    Forțăm model.train() temporar pentru a obține loss_dict, dar folosim torch.no_grad().
    Returnează float('nan') dacă nu s-a putut calcula niciun loss.
    """
    was_training = model.training
    model.train()  # ensure model returns loss_dict with targets

    running_loss = 0.0
    counted_batches = 0

    with torch.no_grad():  # don't compute gradients
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

            try:
                loss_dict = model(images, targets)
            except Exception as e:
                print("Warning: model forward error during validation:", e)
                continue

            if isinstance(loss_dict, list):
                # model returned detections, can't compute loss for this batch
                continue

            losses = sum(loss for loss in loss_dict.values())
            if not torch.isfinite(losses):
                print("Warning: non-finite validation loss; skipping batch.")
                continue

            running_loss += losses.item()
            counted_batches += 1

    # restore previous mode
    if not was_training:
        model.eval()

    if counted_batches == 0:
        return float('nan')

    return running_loss / counted_batches

def main():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dataset = RatsDataset(IMAGES_DIR, CSV_PATH, transforms=ComposeTransforms(hflip_prob=0.5))
    n = len(dataset)
    if n == 0:
        raise SystemExit("Nu s-au gasit imagini/etichete in dataset.")

    # train/val split
    val_size = max(1, int(n * VAL_SPLIT))
    train_size = n - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Dataset size: {n} images, train={train_size}, val={val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # num_classes = background + rat (1) => 2
    num_classes = 2
    model = get_model(num_classes)
    model.to(device)

    # optimizer and lr scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    best_val_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1)
        val_loss = evaluate_on_loader(model, val_loader, device)
        if math.isnan(val_loss):
            print("Val loss could not be computed (no valid batches). Skipping checkpoint/early save.")
        else:
            print(f"Epoch {epoch} --> Train loss {train_loss:.6f}, Val loss {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(OUTPUT_MODEL_DIR, "model_best.pth")
                torch.save(model.state_dict(), best_path)
                print("Saved best model to", best_path)

        lr_scheduler.step()

        # periodic save
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            path = os.path.join(OUTPUT_MODEL_DIR, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), path)
            print("Saved checkpoint:", path)

    # save final artifact
    final_state = os.path.join(OUTPUT_MODEL_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_state)
    print("Training complete. Saved:", final_state)

if __name__ == "__main__":
    main()
