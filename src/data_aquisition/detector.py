# detector.py  (PyTorch .pth loader)
import os
from typing import Optional, Dict, Tuple
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class RatDetector:
    """
    Loader/inferență pentru un model Faster R-CNN salvat în PyTorch (.pth).
    Acceptă atât model complet (torch.save(model)) cât și state_dict (torch.save(model.state_dict())).
    predict(frame_bgr) -> None or {"prob": float, "bbox": (x,y,w,h)} in frame pixels.
    """
    def __init__(self,
                 model_path: str,
                 input_size: int = 256,
                 class_threshold: float = 0.95,
                 device: Optional[str] = None,
                 num_classes: int = 2):
        """
        model_path: calea către .pth
        input_size: dimensiunea la care se redimensionează frame-ul înainte de inferență (256 în cazul tău)
        class_threshold: pragul de încredere pentru a returna o detecție
        device: ex "/cpu:0" sau "cuda:0" sau None => folosește torch.device logic
        num_classes: numărul de clase (inclusiv background). Implicit 2 = background + sobolan
        """
        self.model_path = model_path
        self.input_size = int(input_size)
        self.class_threshold = float(class_threshold)

        # decide device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # așteaptă string ca "cuda:0" sau "cpu"
            self.device = torch.device(device)

        # load
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path!r}")

        # încercăm mai întâi să încărcăm cu map_location
        loaded = torch.load(self.model_path, map_location=self.device)

        # Dacă utilizator a salvat întregul model (nn.Module)
        if isinstance(loaded, torch.nn.Module):
            self.model = loaded.to(self.device)
        else:
            # dacă loaded pare a fi state_dict (mapping tensor)
            try:
                # Construim același tip de model pe care l-ai antrenat:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
                # înlocuim predictorul de clasă cu unul nou (num_classes)
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
                model.load_state_dict(loaded)
                self.model = model.to(self.device)
            except Exception as e:
                # fallback: poate fi dict de checkpoint complex
                # încercăm să detectăm key "model_state_dict" sau "state_dict"
                if isinstance(loaded, dict):
                    if "model_state_dict" in loaded:
                        state = loaded["model_state_dict"]
                    elif "state_dict" in loaded:
                        state = loaded["state_dict"]
                    else:
                        raise RuntimeError("Fișier .pth necunoscut: nu pare nici model complet, nici state_dict.") from e

                    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
                    in_features = model.roi_heads.box_predictor.cls_score.in_features
                    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
                    model.load_state_dict(state)
                    self.model = model.to(self.device)
                else:
                    raise

        self.model.eval()

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """
        Preprocesare:
         - BGR -> RGB
         - resize -> (input_size,input_size)
         - to tensor float32 /255
        Returnează tensor (C,H,W) pe device și dimensiunile originale (h,w).
        """
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        # original dims
        fh, fw = frame_bgr.shape[:2]

        # Convert BGR->RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Resize to square input_size x input_size (same ca la antrenament)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        # to tensor (C,H,W) normalized 0..1
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0).to(self.device)

        return tensor, fh, fw

    def predict(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """
        Rulează inferența pe frameul dat (BGR numpy).
        Returnează None sau dict {"prob": float, "bbox": (x,y,w,h)} în pixeli frame.
        """
        if frame_bgr is None:
            return None

        try:
            tensor, fh, fw = self._preprocess(frame_bgr)
        except Exception as e:
            print("Error in preprocessing:", e)
            return None

        with torch.no_grad():
            try:
                outputs = self.model([tensor])
            except Exception as e:
                print("Model forward error:", e)
                return None

        if not outputs or len(outputs) == 0:
            return None

        out = outputs[0]
        boxes = out.get("boxes")
        scores = out.get("scores")
        labels = out.get("labels", None)

        if boxes is None or scores is None:
            return None

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        if boxes.shape[0] == 0:
            return None

        # alegem cel mai bun (score maxim). Poți înlocui cu altă logică (ex top-k).
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < self.class_threshold:
            return None

        box = boxes[best_idx]  # x1,y1,x2,y2 (aceste coordonate sunt pe imaginea REDIMENSIONATĂ la input_size)

        # mapăm coordonatele din dimensiunea input_size în dimensiunea frame-ului original
        scale_x = fw / float(self.input_size)
        scale_y = fh / float(self.input_size)

        x1 = box[0] * scale_x
        y1 = box[1] * scale_y
        x2 = box[2] * scale_x
        y2 = box[3] * scale_y

        # convert to top-left + width/height, clamp
        x = int(max(0, min(x1, fw - 1)))
        y = int(max(0, min(y1, fh - 1)))
        w = int(max(1, min(x2 - x1, fw - x)))
        h = int(max(1, min(y2 - y1, fh - y)))

        return {"prob": best_score, "bbox": (x, y, w, h)}
