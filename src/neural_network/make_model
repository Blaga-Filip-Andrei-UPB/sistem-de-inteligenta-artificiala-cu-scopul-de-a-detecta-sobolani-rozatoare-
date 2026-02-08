from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

SAVE_DIR = Path("path/to/folder")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = SAVE_DIR / "untrained_model.pth"

class SimpleDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # tiny conv backbone -> produces fixed 1x1 spatial feature
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

        # classification head -> logits per image
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_classes)
        )

        # bbox head -> 4 coordinates per image
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )

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
            x = img.unsqueeze(0)                     # 1,C,H,W
            feats = self.backbone(x)                 # 1,feat,1,1
            cls_logits = self.cls_head(feats)        # 1,num_classes
            box_pred = self.box_head(feats)          # 1,4
            cls_logits_list.append(cls_logits)
            box_pred_list.append(box_pred)

        cls_logits = torch.cat(cls_logits_list, dim=0)  # N,num_classes
        box_pred = torch.cat(box_pred_list, dim=0)      # N,4

        if targets is not None:
            # simple loss computation: use first label/box per image if present
            total_cls_loss = torch.tensor(0.0, device=cls_logits.device)
            total_bbox_loss = torch.tensor(0.0, device=box_pred.device)
            count_cls = 0
            count_bbox = 0

            for i, tgt in enumerate(targets):
                if tgt is None:
                    continue

                if "labels" in tgt and tgt["labels"].numel() > 0:
                    gt_label = tgt["labels"][0].to(cls_logits.device).long().unsqueeze(0)
                    logits = cls_logits[i].unsqueeze(0)
                    total_cls_loss = total_cls_loss + F.cross_entropy(logits, gt_label)
                    count_cls += 1

                if "boxes" in tgt and tgt["boxes"].numel() > 0:
                    gt_box = tgt["boxes"][0].to(box_pred.device).unsqueeze(0)
                    pred_box = box_pred[i].unsqueeze(0)
                    total_bbox_loss = total_bbox_loss + F.mse_loss(pred_box, gt_box)
                    count_bbox += 1

            if count_cls > 0:
                total_cls_loss = total_cls_loss / count_cls
            else:
                total_cls_loss = torch.tensor(0.0, device=cls_logits.device)

            if count_bbox > 0:
                total_bbox_loss = total_bbox_loss / count_bbox
            else:
                total_bbox_loss = torch.tensor(0.0, device=box_pred.device)

            return {"loss_classifier": total_cls_loss, "loss_box_reg": total_bbox_loss}

        # inference path: produce one detection per image
        probs = F.softmax(cls_logits, dim=1)
        scores, labels = torch.max(probs, dim=1)

        x_min = torch.min(box_pred[:, 0], box_pred[:, 2]).unsqueeze(1)
        x_max = torch.max(box_pred[:, 0], box_pred[:, 2]).unsqueeze(1)
        y_min = torch.min(box_pred[:, 1], box_pred[:, 3]).unsqueeze(1)
        y_max = torch.max(box_pred[:, 1], box_pred[:, 3]).unsqueeze(1)
        boxes = torch.clamp(torch.cat([x_min, y_min, x_max, y_max], dim=1), min=0.0)

        detections = []
        for i in range(len(imgs)):
            detections.append({
                "boxes": boxes[i].unsqueeze(0),
                "labels": labels[i].unsqueeze(0),
                "scores": scores[i].unsqueeze(0)
            })
        return detections


def build_and_save_state(save_path=STATE_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleDetector(num_classes=2)
    model.to(device)

    # initialize weights for conv/linear layers
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)

    model.apply(init_weights)

    # save only the learned parameter tensors
    torch.save(model.state_dict(), str(save_path))

    print(f"[{datetime.datetime.now().isoformat()}] Saved state_dict -> {save_path}")
    return model


if __name__ == "__main__":
    build_and_save_state()
