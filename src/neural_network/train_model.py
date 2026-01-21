"""
Script de antrenare Faster R-CNN de la zero pentru proiectul de detectare rozatoare.
Salvează:
 - models/trained_model.pt
 - results/training_history.csv
 - results/test_metrics.json
 - docs/etapa5_antrenare_model.md
"""

import os
import random
import math
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

# ----------------- CONFIG -----------------
IMAGES_DIR = r"path/to/dir"
CSV_PATH   = r"path/to/file"

MODELS_DIR = r"path/to/dir"
RESULTS_DIR = r"path/to/dir"
DOCS_DIR = r"path/to/dir"

NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 8
LR_GAMMA = 0.1
VAL_SPLIT = 0.2
RANDOM_SEED = 42
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]
SCORE_THRESHOLD = 0.5  # prag scor predictii la evaluare
IOU_THRESHOLD = 0.5    # prag IoU pentru a considera o detectie ca TP
# ------------------------------------------

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Etichetare: toate clasele relevante vor fi unificate ca "rat" -> id = 1
def map_label_to_id(label_name):
    return 1

class RatsDataset(Dataset):
    """
    Dataset care:
     - cauta fisiere in folderul de imagini
     - citeste CSV-ul cu bounding boxes
     - returneaza (image_tensor, target_dict) compatibil Faster R-CNN
    """
    def __init__(self, images_dir, csv_path, transforms=None):
        self.images_dir = Path(images_dir)
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = pd.DataFrame(columns=['label_name','image_name','bbox_x','bbox_y','bbox_width','bbox_height','image_width','image_height'])

        files = []
        if self.images_dir.exists():
            for p in self.images_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    files.append(p.name)

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
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

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
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h

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
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(map_label_to_id(r['label_name']))
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

        img_tensor = F.to_tensor(img)
        return img_tensor, target

class ComposeTransforms:
    """
    Transformari simple: flip or nu; pastrate in format PIL -> tensor la final.
    """
    def __init__(self, hflip_prob=0.5):
        self.hflip_prob = hflip_prob

    def __call__(self, image, target):
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

def init_weights(module):
    """
    Inițializări recomandate:
     - Conv2d: kaiming normal
     - Linear: normal (mu=0, sigma=0.01)
     - BatchNorm/GroupNorm: weight=1, bias=0
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        if getattr(module, "weight", None) is not None:
            nn.init.constant_(module.weight, 1.0)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias, 0.0)

def get_model_from_scratch(num_classes):
    """
    Construiește Faster R-CNN cu ResNet50-FPN, fără greutăți pre-antrenate.
    Înlocuiește predictorul pentru a avea num_classes și re-initializează parametrii.
    """
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    except TypeError:
        # compatibilitate cu versiuni vechi torchvision
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.apply(init_weights)
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    running_loss = 0.0
    n_batches = len(data_loader)
    start = time.time()
    for i, (images, targets) in enumerate(data_loader, 1):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        if i % print_freq == 0 or i == n_batches:
            elapsed = time.time() - start
            print(f"Epoch {epoch} [{i}/{n_batches}] Loss: {losses.item():.4f} Elapsed: {elapsed:.1f}s")
    avg_loss = running_loss / n_batches
    print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_on_loader(model, data_loader, device):
    """
    Evaluare simplificată:
     - pentru fiecare imagine: se extrag predictiile cu score >= SCORE_THRESHOLD
     - se face potrivire greedy între GT si predictii după IoU >= IOU_THRESHOLD
     - se numără TP, FP, FN -> se calculează precision, recall, f1
     - accuracy este calculată la nivel de imagine: (numar imagini corect clasificate ca prezenta/absenta) / N
    """
    model.eval()
    total_TP = 0
    total_FP = 0
    total_FN = 0
    correct_images = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)  # list of dicts with 'boxes', 'labels', 'scores'
            for out, t in zip(outputs, targets):
                total_images += 1
                gt_boxes = t["boxes"].cpu().numpy() if t["boxes"].numel() > 0 else np.zeros((0,4))
                pred_boxes = out.get("boxes", torch.zeros((0,4))).cpu().numpy()
                pred_scores = out.get("scores", torch.zeros((0,))).cpu().numpy()
                # filter by score
                keep_idx = np.where(pred_scores >= SCORE_THRESHOLD)[0]
                pred_boxes = pred_boxes[keep_idx]

                # match preds to GT using greedy IoU
                used_gt = set()
                TP = 0
                for pb in pred_boxes:
                    ious = [iou(pb, gb) for gb in gt_boxes] if len(gt_boxes) > 0 else []
                    if len(ious) == 0:
                        # no GT -> false positive
                        continue
                    best_idx = int(np.argmax(ious))
                    if best_idx in used_gt:
                        # gt already matched
                        continue
                    if ious[best_idx] >= IOU_THRESHOLD:
                        TP += 1
                        used_gt.add(best_idx)

                FP = len(pred_boxes) - TP
                FN = len(gt_boxes) - TP

                total_TP += TP
                total_FP += FP
                total_FN += FN

                # image-level correctness: consider corectă dacă există GT și model a detectat >=1 box matched,
                # sau dacă nu există GT și model a returnat 0 predictii (după prag).
                has_gt = len(gt_boxes) > 0
                has_pred = len(pred_boxes) > 0
                image_correct = (has_gt and TP > 0) or (not has_gt and not has_pred)
                if image_correct:
                    correct_images += 1

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = correct_images / total_images if total_images > 0 else 0.0

    return {
        "TP": int(total_TP),
        "FP": int(total_FP),
        "FN": int(total_FN),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy)
    }

def iou(boxA, boxB):
    """
    Compute IoU between two boxes in format [x1,y1,x2,y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def main():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dataset = RatsDataset(IMAGES_DIR, CSV_PATH, transforms=ComposeTransforms(hflip_prob=0.5))
    n = len(dataset)
    if n == 0:
        raise SystemExit("Nu s-au gasit imagini/etichete în dataset. Verifică IMAGES_DIR și CSV_PATH.")

    val_size = max(1, int(n * VAL_SPLIT))
    train_size = n - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Dataset size: {n} images, train={train_size}, val={val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    num_classes = 2  # background + rat
    model = get_model_from_scratch(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    history_path = os.path.join(RESULTS_DIR, "training_history.csv")
    # create CSV header
    with open(history_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    best_val_loss = float('inf')
    best_path = os.path.join(MODELS_DIR, "trained_model_best.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        val_loss = evaluate_loss_on_loader_for_checkpoint(model, val_loader, device)
        if math.isnan(val_loss):
            print("Val loss could not be computed for any batch.")
            val_loss_to_write = "nan"
        else:
            val_loss_to_write = f"{val_loss:.6f}"

        with open(history_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss_to_write}\n")

        print(f"Epoch {epoch} summary: train_loss={train_loss:.6f}, val_loss={val_loss_to_write}")

        # save best
        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print("Saved new best model:", best_path)

        lr_scheduler.step()

        # periodic checkpoint
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            ckpt_path = os.path.join(MODELS_DIR, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print("Saved checkpoint:", ckpt_path)

    final_state = os.path.join(MODELS_DIR, "trained_model.pt")
    torch.save(model.state_dict(), final_state)
    print("Final model saved to:", final_state)

    # Evaluate final model on validation set and save metrics
    metrics = evaluate_on_loader(model, val_loader, device)
    metrics_out = {
        "test_accuracy": round(metrics["accuracy"], 4),
        "test_f1_macro": round(metrics["f1"], 4),
        "test_precision_macro": round(metrics["precision"], 4),
        "test_recall_macro": round(metrics["recall"], 4),
        "TP": metrics["TP"],
        "FP": metrics["FP"],
        "FN": metrics["FN"],
        "iou_threshold": IOU_THRESHOLD,
        "score_threshold": SCORE_THRESHOLD,
        "num_test_images": len(val_loader.dataset)
    }

    metrics_path = os.path.join(RESULTS_DIR, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=4)
    print("Saved metrics to:", metrics_path)

    # Write documentation file with hyperparameters + justifications + metrics + error analysis
    md_path = os.path.join(DOCS_DIR, "etapa5_antrenare_model.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_documentation(metrics_out))
    print("Documentation written to:", md_path)

def evaluate_loss_on_loader_for_checkpoint(model, data_loader, device):
    """
    Funcție simplă pentru a calcula loss mediu pe loader (folosit pentru decizii de checkpoint).
    Folosește model.train() pentru a obține loss_dict, dar cu torch.no_grad() pentru a nu acumula gradienți.
    """
    was_training = model.training
    model.train()
    running_loss = 0.0
    counted = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            try:
                loss_dict = model(images, targets)
            except Exception as e:
                print("Warning: forward error during val loss computation:", e)
                continue
            if isinstance(loss_dict, list):
                continue
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()
            counted += 1
    if not was_training:
        model.eval()
    if counted == 0:
        return float('nan')
    return running_loss / counted

def generate_documentation(metrics):
    """
    Generează conținutul pentru docs/etapa5_antrenare_model.md (română).
    Include tabel hiperparametri + justificări, metrici finale și analiza erorilor (4 paragrafe).
    """
    md = []
    md.append("# Etapa 5 — Antrenare model\n")
    md.append("## 1. Tabel hiperparametri și justificări\n")
    md.append("| Hiperparametru | Valoare | Justificare |\n")
    md.append("|---|---:|---|\n")
    md.append(f"| num_epochs | {NUM_EPOCHS} | {NUM_EPOCHS} epoci pentru a permite convergența când se antrenează de la zero; poate necesita creștere în funcție de mărimea setului de date |\n")
    md.append(f"| batch_size | {BATCH_SIZE} | compromis între stabilitatea gradientului și memoria GPU; ajustați în funcție de memorie |\n")
    md.append(f"| learning_rate | {LEARNING_RATE} | LR inițial moderat; pentru antrenare de la zero este acceptabil, dar se poate crește sau folosi un scheduler mai sofisticat |\n")
    md.append(f"| weight_decay | {WEIGHT_DECAY} | regularizare L2 pentru a reduce overfitting |\n")
    md.append(f"| lr_step_size | {LR_STEP_SIZE} | scădere programată a LR pentru stabilizare |\n")
    md.append(f"| lr_gamma | {LR_GAMMA} | factor de reducere a LR la pași |\n")
    md.append(f"| val_split | {VAL_SPLIT} | proporție pentru validare, folosită și pentru metrici finale |\n")
    md.append(f"| random_seed | {RANDOM_SEED} | reproducibilitate |\n")
    md.append(f"| score_threshold (eval) | {SCORE_THRESHOLD} | prag scor pentru considerarea predicțiilor la evaluare |\n")
    md.append(f"| iou_threshold (eval) | {IOU_THRESHOLD} | prag IoU pentru a considera o predicție ca adevărat pozitiv |\n")

    md.append("\n## 2. Metrici pe setul de test (val)\n")
    md.append("Metricile de mai jos sunt calculate după potrivirea predicțiilor cu ground-truth folosind IoU >= " + str(IOU_THRESHOLD) + " și prag de scor " + str(SCORE_THRESHOLD) + ".\n\n")
    md.append("```json\n")
    md.append(json.dumps(metrics, indent=4))
    md.append("\n```\n")

    md.append("## 3. (Nivel 2) Analiză erori în context industrial\n")
    md.append("1. Într-un mediu industrial, fals-pozitivele (FP) pot cauza acțiuni inutile: declanșarea unei alerte, trimiterea unei echipe de inspecție sau activarea unor sisteme mecanice. Dacă FP este ridicat, costurile operaționale cresc și încrederea operatorilor scade. Prin urmare, este esențial să se prioritizeze praguri de încredere stricte și validări ulterioare (ex: confirmare cu o cameră secundară) înainte de a lua măsuri costisitoare.\n\n")
    md.append("2. Fals-negativele (FN) pot avea consecințe grave: o infestare neraportată poate duce la daune, contaminare sau riscuri pentru sănătate. În aplicații critice trebuie setat un prag de detecție mai agresiv sau un flux secundar de verificare (de exemplu analiza periodică manuală a zonelor cu risc) pentru a reduce FN, chiar dacă asta înseamnă acceptarea unui număr controlat de FP.\n\n")
    md.append("3. Erorile de localizare (IoU scăzut) pot determina corectitudinea acțiunilor mecanice: de exemplu, dacă un sistem automat încearcă să captureze/îndepărteze rozătoarele, o localizare inexactă poate face manevra ineficientă sau periculoasă. Dacă aplicația include acțiuni fizice, se recomandă o etapă de planificare care să țină cont de incertitudinea localizărilor (zone tampon, acțiuni conservatoare).\n\n")
    md.append("4. Robustetea în fața variației de iluminare și a mediului: mediile industriale au variații mari de iluminare, obiecte similare ca formă și ocazional obstrucții. Modelele antrenate de la zero sunt sensibile la aceste variații fără augmentări extinse. Se recomandă pipeline de augmentări (resize aleator, variații de contrast, adăugare de zgomot) și testare în condiții reale înainte de implementare la scară.\n\n")

    md.append("## 4. Observații practice\n")
    md.append("- Fișierele generate:\n")
    md.append(f"  - Model: {os.path.join('models','trained_model.pt')}\n")
    md.append(f"  - Istoric antrenare: {os.path.join('results','training_history.csv')}\n")
    md.append(f"  - Metrici: {os.path.join('results','test_metrics.json')}\n")
    md.append("- Definițiile metricilor sunt explicate în secțiunea 2.\n")

    return "".join(md)

if __name__ == "__main__":
    main()
