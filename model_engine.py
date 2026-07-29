import os
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["Hồng Trung Quốc", "Hồng Lạng Sơn", "Hồng Đà Lạt"]
CLASS_DIR_MAP = {
    "hong_trung_quoc": "Hồng Trung Quốc",
    "hong_lang_son": "Hồng Lạng Sơn",
    "hong_da_lat": "Hồng Đà Lạt"
}

def preprocess_image_tensor(pil_img):
    """
    Standard ImageNet Preprocessing: Ensure RGB mode, resize to 224x224, 
    normalize with ImageNet mean & std.
    """
    pil_img = pil_img.convert('RGB')
    img_resized = pil_img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    
    arr = arr.transpose((2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
    return tensor

def augment_image(pil_img):
    """
    SOTA Online Data Augmentation Engine:
    1. Random Horizontal Flip
    2. Random Rotation (-25 to +25 degrees)
    3. Random Color, Contrast & Brightness Jitter
    """
    img = pil_img.copy()
    
    # 1. Random Horizontal Flip (50% chance)
    if np.random.rand() > 0.5:
        img = ImageOps.mirror(img)
        
    # 2. Random Rotation (-25 to +25 degrees)
    angle = np.random.uniform(-25, 25)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    
    # 3. Random Color & Brightness Jitter
    if np.random.rand() > 0.5:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(np.random.uniform(0.75, 1.25))
    if np.random.rand() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(np.random.uniform(0.8, 1.2))
        
    return img

class PersimmonDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for efficient mini-batch processing and augmentation.
    """
    def __init__(self, image_paths, labels, is_train=False):
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        raw_pil = Image.open(path).convert('RGB')
        if self.is_train:
            raw_pil = augment_image(raw_pil)
        
        img_resized = raw_pil.resize((224, 224), Image.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose((2, 0, 1))
        
        return torch.from_numpy(arr), torch.tensor(label, dtype=torch.long)

class PersimmonClassifier:
    """
    State-of-the-Art (SOTA) Transfer Learning Persimmon Classifier.
    Features:
    - Supported Backbones: ConvNeXt-Tiny, EfficientNetV2-S, ResNet-50, MobileNetV3-Large.
    - Two-Stage Progressive Fine-Tuning (Head Warmup -> Deep Unfreezing with Differential LR).
    - Test-Time Augmentation (TTA) multi-view inference.
    - Automatic Mixed Precision (AMP) GPU acceleration.
    - Detailed Validation Metrics (Accuracy, Precision, Recall, Macro F1-Score).
    """
    def __init__(self, model_type="efficientnet_v2", weights_path="models/persimmon_model.pth"):
        self.model_type = model_type
        self.weights_path = weights_path
        self.num_classes = len(CLASSES)
        self.is_custom_trained = False
        self.model = self._build_model()
        self.load_weights_if_exists()

    def _build_model(self):
        """Build transfer learning model with frozen backbone + regularized classifier head"""
        if self.model_type == "convnext_tiny":
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Sequential(
                nn.Dropout(0.35),
                nn.Linear(in_features, self.num_classes)
            )
        elif self.model_type == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.35),
                nn.Linear(in_features, self.num_classes)
            )
        elif self.model_type == "mobilenet_v3":
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Sequential(
                nn.Dropout(0.35),
                nn.Linear(in_features, self.num_classes)
            )
        else: # Default: EfficientNetV2-S
            model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Sequential(
                nn.Dropout(0.35),
                nn.Linear(in_features, self.num_classes)
            )
            
        model = model.to(DEVICE)
        model.eval()
        return model

    def unfreeze_deep_layers(self):
        """
        Stage 2 Unfreezing: Unfreeze deepest backbone blocks for domain-specific feature tuning.
        """
        if self.model_type == "convnext_tiny":
            for param in self.model.features[6:].parameters():
                param.requires_grad = True
        elif self.model_type == "resnet50":
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        elif self.model_type == "mobilenet_v3":
            for param in self.model.features[12:].parameters():
                param.requires_grad = True
        else: # EfficientNetV2-S
            for param in self.model.features[6:].parameters():
                param.requires_grad = True
        print(f"[ModelEngine] Stage 2: Deep feature layers unfrozen for fine-grained learning.")

    def load_weights_if_exists(self):
        if os.path.exists(self.weights_path):
            try:
                checkpoint = torch.load(self.weights_path, map_location=DEVICE)
                saved_arch = checkpoint.get('model_type', self.model_type)
                if saved_arch != self.model_type:
                    print(f"[ModelEngine] Checkpoint architecture [{saved_arch}] differs from current [{self.model_type}]. Initializing fresh backbone.")
                    return
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.is_custom_trained = True
                print(f"[ModelEngine] Loaded fine-tuned weights for [{self.model_type.upper()}] from {self.weights_path}")
            except Exception as e:
                print(f"[ModelEngine] Warning loading weights: {e}")

    def save_weights(self):
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'classes': CLASSES
        }, self.weights_path)
        self.is_custom_trained = True
        print(f"[ModelEngine] Saved best model weights to {self.weights_path}")

    def predict(self, image_path):
        """
        SOTA Prediction Engine featuring Test-Time Augmentation (TTA).
        Runs dual-pass inference (Original + Mirrored) and averages softmax distributions.
        """
        try:
            orig_img = Image.open(image_path).convert('RGB')
            tensor_orig = preprocess_image_tensor(orig_img)
            # Test-Time Augmentation (TTA): Horizontally flipped view
            tensor_flip = preprocess_image_tensor(ImageOps.mirror(orig_img))
        except Exception as e:
            return None, f"Lỗi đọc ảnh: {str(e)}"

        # Morphological & Color Analysis metadata
        img_np = np.array(orig_img)
        h, w, _ = img_np.shape
        aspect_ratio = w / float(h)
        
        avg_r = np.mean(img_np[:, :, 0])
        avg_g = np.mean(img_np[:, :, 1])
        avg_b = np.mean(img_np[:, :, 2])
        redness = avg_r / (avg_g + avg_b + 1e-5)

        self.model.eval()
        with torch.no_grad():
            # Pass 1: Original View
            out_orig = self.model(tensor_orig)
            prob_orig = torch.softmax(out_orig / 1.2, dim=1)
            
            # Pass 2: TTA Mirrored View
            out_flip = self.model(tensor_flip)
            prob_flip = torch.softmax(out_flip / 1.2, dim=1)
            
            # TTA Ensemble Average
            probs = ((prob_orig + prob_flip) / 2.0).cpu().numpy()[0]

        if not self.is_custom_trained:
            # Physical domain indicators fallback when untrained
            adj_scores = probs.copy()
            if redness > 0.90:
                adj_scores[0] += 0.3 # Hồng Trung Quốc màu cam đỏ đậm hơn
            if aspect_ratio < 0.96:
                adj_scores[1] += 0.3 # Hồng Lạng Sơn hình thuôn dài
            elif 0.96 <= aspect_ratio <= 1.08:
                adj_scores[2] += 0.3 # Hồng Đà Lạt hình oval cân đối

            exp_s = np.exp(adj_scores)
            probs = exp_s / exp_s.sum()

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])

        results = {
            "predicted_class": CLASSES[top_idx],
            "confidence": round(confidence * 100, 2),
            "is_custom_trained": self.is_custom_trained,
            "model_architecture": self.model_type.upper(),
            "tta_enabled": True,
            "probabilities": {
                CLASSES[0]: round(float(probs[0]) * 100, 2),
                CLASSES[1]: round(float(probs[1]) * 100, 2),
                CLASSES[2]: round(float(probs[2]) * 100, 2)
            },
            "visual_analysis": {
                "aspect_ratio": round(aspect_ratio, 2),
                "shape_type": "Hình bẹt / vuông" if aspect_ratio > 1.05 else ("Hình thuôn / thoi" if aspect_ratio < 0.96 else "Hình oval / quả trứng"),
                "color_intensity": "Cam đỏ đậm" if redness > 0.9 else ("Vàng cam" if redness > 0.75 else "Vàng xanh / Tự nhiên"),
                "redness_index": round(float(redness), 2)
            }
        }
        return results, None

    def train_on_dataset(self, dataset_dir, epochs=10, lr=0.001):
        """
        SOTA Transfer Learning Fine-Tuning Pipeline:
        1. Stratified Dataset Split (80% Train, 20% Val).
        2. Two-Stage Progressive Fine-Tuning:
           - Stage 1 (Head Warmup): Train classifier head with backbone frozen.
           - Stage 2 (Deep Tuning): Unfreeze deep vision blocks with differential LR (lr * 0.1).
        3. Automatic Mixed Precision (AMP) GPU acceleration.
        4. Detailed Validation Metrics (Accuracy, Macro F1-Score).
        5. Best Checkpoint Preservation & In-Memory Reload.
        """
        image_paths = []
        labels = []
        
        dir_to_label = {
            "hong_trung_quoc": 0,
            "hong_lang_son": 1,
            "hong_da_lat": 2
        }

        for sub_dir, lbl_idx in dir_to_label.items():
            path = os.path.join(dataset_dir, sub_dir)
            if os.path.exists(path):
                for f in os.listdir(path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        image_paths.append(os.path.join(path, f))
                        labels.append(lbl_idx)

        if len(image_paths) < 3:
            return False, "Cần ít nhất 3 ảnh trong các thư mục lớp để huấn luyện."

        total_samples = len(image_paths)
        
        # --- STRATIFIED TRAIN / VALIDATION SPLIT ---
        class_indices = {0: [], 1: [], 2: []}
        for idx, lbl in enumerate(labels):
            class_indices[lbl].append(idx)

        train_idx = []
        val_idx = []

        if total_samples >= 5:
            for lbl, idxs in class_indices.items():
                if not idxs:
                    continue
                shuffled = np.random.permutation(idxs)
                v_count = max(1, int(len(shuffled) * 0.2)) if len(shuffled) >= 3 else (1 if len(shuffled) > 1 else 0)
                val_idx.extend(shuffled[:v_count])
                train_idx.extend(shuffled[v_count:])
            if not train_idx:
                train_idx = val_idx.copy()
        else:
            train_idx = list(range(total_samples))
            val_idx = list(range(total_samples))

        train_paths = [image_paths[i] for i in train_idx]
        train_lbls = [labels[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        val_lbls = [labels[i] for i in val_idx]

        train_dataset = PersimmonDataset(train_paths, train_lbls, is_train=True)
        val_dataset = PersimmonDataset(val_paths, val_lbls, is_train=False)

        batch_size = min(8, max(1, len(train_paths)))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Determine Two-Stage Progressive Fine-Tuning Split
        stage1_epochs = max(1, int(epochs * 0.4)) if epochs >= 4 else epochs
        
        # Stage 1 Optimizer (Head Only)
        head_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs)
        
        # AMP Scaler for GPU acceleration
        use_amp = DEVICE.type == 'cuda'
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_loss = float('inf')
        history = []
        is_stage2 = False

        for epoch in range(epochs):
            # Transition to Stage 2: Unfreeze Deep Feature Layers
            if epoch >= stage1_epochs and not is_stage2 and epochs >= 4:
                self.unfreeze_deep_layers()
                is_stage2 = True
                
                # Group parameters with differential learning rate (Backbone deep: lr * 0.1, Head: lr)
                deep_params = []
                head_params = []
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if 'classifier' in name or 'fc' in name:
                            head_params.append(param)
                        else:
                            deep_params.append(param)
                            
                optimizer = torch.optim.AdamW([
                    {'params': deep_params, 'lr': lr * 0.1},
                    {'params': head_params, 'lr': lr}
                ], weight_decay=1e-2)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - stage1_epochs))

            # --- TRAINING PHASE ---
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        out = self.model(imgs)
                        loss = criterion(out, lbls)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = self.model(imgs)
                    loss = criterion(out, lbls)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(out, dim=1)
                train_correct += (preds == lbls).sum().item()
                train_total += imgs.size(0)

            scheduler.step()
            train_acc = round(100.0 * train_correct / max(train_total, 1), 2)
            avg_train_loss = round(train_loss / max(train_total, 1), 4)

            # --- VALIDATION PHASE & METRICS (ACCURACY, PRECISION, RECALL, F1) ---
            self.model.eval()
            val_loss = 0.0
            val_preds_all = []
            val_targets_all = []
            
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            out = self.model(imgs)
                            v_loss = criterion(out, lbls)
                    else:
                        out = self.model(imgs)
                        v_loss = criterion(out, lbls)
                        
                    val_loss += v_loss.item() * imgs.size(0)
                    preds = torch.argmax(out, dim=1)
                    val_preds_all.extend(preds.cpu().numpy())
                    val_targets_all.extend(lbls.cpu().numpy())

            val_total = max(len(val_targets_all), 1)
            val_correct = sum(p == t for p, t in zip(val_preds_all, val_targets_all))
            avg_val_loss = round(val_loss / val_total, 4)
            val_acc = round(100.0 * val_correct / val_total, 2)

            # Calculate Macro F1-Score
            f1_scores = []
            for c in range(self.num_classes):
                tp = sum((p == c and t == c) for p, t in zip(val_preds_all, val_targets_all))
                fp = sum((p == c and t != c) for p, t in zip(val_preds_all, val_targets_all))
                fn = sum((p != c and t == c) for p, t in zip(val_preds_all, val_targets_all))
                prec = tp / max(tp + fp, 1e-5)
                rec = tp / max(tp + fn, 1e-5)
                f1 = 2 * prec * rec / max(prec + rec, 1e-5)
                f1_scores.append(f1)
            macro_f1 = round(float(np.mean(f1_scores)) * 100, 2)

            history.append({
                "epoch": epoch + 1,
                "stage": 2 if is_stage2 else 1,
                "loss": avg_train_loss,
                "accuracy": train_acc,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc,
                "val_f1_score": macro_f1
            })
            print(f"[Training] Epoch {epoch+1}/{epochs} (Stage {2 if is_stage2 else 1}) | Train Loss: {avg_train_loss} Acc: {train_acc}% | Val Loss: {avg_val_loss} Acc: {val_acc}% F1: {macro_f1}%")

            # --- SAVE BEST MODEL CHECKPOINT ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_weights()

        # Reload best weights into memory RAM
        if best_val_loss < float('inf'):
            self.load_weights_if_exists()
            
        self.model.eval()
        return True, history
