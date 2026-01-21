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
import torchvision.transforms.functional as F

# ---------- USER CONFIG ----------
IMAGES_DIR = r"path/to/directory"
CSV_PATH   = r"path/to/file"
OUTPUT_MODEL_DIR = r"path/to/directory"

NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 8
LR_GAMMA = 0.1
VAL_SPLIT = 0.2
RANDOM_SEED = 42
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]
# ---------------------------------

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

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
        img = Image.open(img_path).convert("RGB")  # convert grayscale->RGB by duplication

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

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
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

        # (Optional) random small translation or scale could be added here

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one (num_classes = 1 class + background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1):
    model.train()
    running_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        if i % print_freq == 0:
            print(f"Epoch {epoch} Iter {i}/{len(data_loader)} Loss: {losses.item():.4f}")
    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_on_loader(model, data_loader, device):
    """
    Calculează loss-ul mediu pe loader-ul de validare.
    Observație: forțăm model.train() temporar pentru a obține loss_dict de la Faster R-CNN,
    dar folosim torch.no_grad() ca să nu acumulăm gradienți.
    Returnează float('nan') dacă nu s-a putut calcula niciun loss.
    """
    was_training = model.training
    model.train()  # -> garantăm că forward-ul returnează loss_dict când primesc targets

    running_loss = 0.0
    counted_batches = 0

    with torch.no_grad():  # nu vrem gradienți în evaluare
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            try:
                loss_dict = model(images, targets)
            except Exception as e:
                # în caz de eroare neașteptată, afișăm și sărim batch-ul
                print("Warning: model forward error during validation:", e)
                continue

            if isinstance(loss_dict, list):
                # model returned detections, can't compute loss for this batch
                continue

            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()
            counted_batches += 1

    # restaurăm modul anterior al modelului
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
            print(f"Epoch {epoch} --> Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")
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

    # save final artifacts
    final_state = os.path.join(OUTPUT_MODEL_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_state)
    # optionally save entire model (larger file, but easier to load)
    full_path = os.path.join(OUTPUT_MODEL_DIR, "model_complete.pth")
    torch.save(model, full_path)
    print("Training complete. Saved:", final_state, "and", full_path)

if __name__ == "__main__":
    main()
