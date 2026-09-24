"""
4_train_model.py
================
Custom PyTorch Object Detection Training Script for Indian Traditional Games.
Built strictly from scratch without any predefined models (No YOLO, No torchvision models).

Four Core Components:
1. Custom Dataset Loader (torch.utils.data.Dataset)
2. Dual-Output Architecture: IndianTraditionalGameDetector (nn.Module)
3. Training Loop with Adam, Combined Loss (CrossEntropyLoss + SmoothL1Loss), and Early Stopping (Patience = 3)
4. Model Persistence: Saves state dict exactly as 'indian_games_detector.pth'
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from lxml import etree
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# Class Label Mapping (0 = Background, 1-5 = Indian Traditional Games)
# ==============================================================================
CLASS_TO_IDX = {
    "background": 0,
    "SpinningTop": 1,
    "SevenStones": 2,
    "Thattangal": 3,
    "Thayam": 4,
    "SnakesandLadders": 5
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


# ==============================================================================
# Component 1: Custom Dataset Loader (torch.utils.data.Dataset)
# ==============================================================================
class IndianGamesDatasetLoader(Dataset):
    """
    Custom Dataset Loader that:
    1. Parses Pascal VOC XML files.
    2. Resizes images to (512, 512) and normalizes pixel values to [0, 1].
    3. Normalizes bounding box coordinates to [0, 1] relative to original image size.
    4. Maps 5 string class labels to integers (0 = background, 1-5 = game classes).
    """
    def __init__(self, annotations_dir, images_dir, target_size=(512, 512)):
        self.annotations_dir = Path(annotations_dir)
        self.images_dir = Path(images_dir)
        self.target_size = target_size
        
        # Load all XML annotation files
        self.xml_files = sorted(list(self.annotations_dir.glob("*.xml")))
        if len(self.xml_files) == 0:
            raise RuntimeError(f"No XML annotation files found in {self.annotations_dir}")

    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, idx):
        xml_path = self.xml_files[idx]
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
        
        # Determine image file path (stem matching with .jpg)
        xml_stem = xml_path.stem
        img_path = self.images_dir / f"{xml_stem}.jpg"
        if not img_path.exists():
            # Fallback to XML filename tag
            filename_elem = root.find("filename")
            filename = filename_elem.text.strip() if filename_elem is not None and filename_elem.text else f"{xml_stem}.jpg"
            if not filename.endswith(".jpg"):
                filename += ".jpg"
            img_path = self.images_dir / filename
            
        # Load image via OpenCV
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
            
        orig_h, orig_w = image.shape[:2]
        size_elem = root.find("size")
        if size_elem is not None:
            w_tag = size_elem.find("width")
            h_tag = size_elem.find("height")
            if w_tag is not None and h_tag is not None:
                orig_w = float(w_tag.text)
                orig_h = float(h_tag.text)

        # Parse objects and bounding boxes
        objs = root.findall("object")
        if len(objs) == 0:
            label = CLASS_TO_IDX["background"]
            bbox = [0.0, 0.0, 1.0, 1.0]
        else:
            # Map string class label to integer
            primary_name = objs[0].find("name").text.strip()
            label = CLASS_TO_IDX.get(primary_name, CLASS_TO_IDX["background"])
            
            # Aggregate spatial extent (enclosing bounding box for multi-object classes like stones)
            xmins, ymins, xmaxs, ymaxs = [], [], [], []
            for obj in objs:
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmins.append(float(bndbox.find("xmin").text))
                    ymins.append(float(bndbox.find("ymin").text))
                    xmaxs.append(float(bndbox.find("xmax").text))
                    ymaxs.append(float(bndbox.find("ymax").text))
                    
            if len(xmins) > 0:
                # Normalize float coordinates to [0, 1]
                xmin_norm = max(0.0, min(1.0, min(xmins) / orig_w))
                ymin_norm = max(0.0, min(1.0, min(ymins) / orig_h))
                xmax_norm = max(0.0, min(1.0, max(xmaxs) / orig_w))
                ymax_norm = max(0.0, min(1.0, max(ymaxs) / orig_h))
                bbox = [xmin_norm, ymin_norm, xmax_norm, ymax_norm]
            else:
                bbox = [0.0, 0.0, 1.0, 1.0]

        # Resize image to target 512x512 and convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to PyTorch float tensor: (H, W, C) -> (C, H, W) normalized to [0, 1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label_tensor = torch.tensor(label, dtype=torch.long)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        
        targets = (label_tensor, bbox_tensor)
        return image_tensor, targets


# ==============================================================================
# Component 2: Dual-Output Architecture (nn.Module)
# ==============================================================================
class IndianTraditionalGameDetector(nn.Module):
    """
    Custom Object Detector with:
    - 4-Block Conv2d Feature Extractor (strictly built from scratch without predefined backbones)
    - 6-Class Linear Classifier Head
    - 4-Coordinate Linear Regressor Head with Sigmoid activation ([0, 1] normalized bounding box)
    """
    def __init__(self, num_classes=6):
        super(IndianTraditionalGameDetector, self).__init__()
        
        # Block 1: Input (3, 512, 512) -> (32, 128, 128)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: (32, 128, 128) -> (64, 32, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: (64, 32, 32) -> (128, 8, 8)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: (128, 8, 8) -> (256, 4, 4)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Spatial pooling to a fixed representation
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        flattened_dim = 256 * 2 * 2  # 1,024 features
        
        # Head 1: 6-Class Linear Classifier Head
        self.classifier_head = nn.Linear(flattened_dim, num_classes)
        
        # Head 2: 4-Coordinate Linear Regressor Head with Sigmoid Activation
        self.regressor_head = nn.Sequential(
            nn.Linear(flattened_dim, 4),
            nn.Sigmoid()  # Keeps coordinate predictions strictly bounded in [0, 1]
        )

    def forward(self, x):
        feat = self.block1(x)
        feat = self.block2(feat)
        feat = self.block3(feat)
        feat = self.block4(feat)
        feat = self.pool(feat)
        
        flattened = torch.flatten(feat, 1)
        
        cls_logits = self.classifier_head(flattened)
        bbox_coords = self.regressor_head(flattened)
        
        return cls_logits, bbox_coords


# ==============================================================================
# Component 3 & 4: Training Loop with Early Stopping & Model Saving
# ==============================================================================
def train_model(
    annotations_dir,
    images_dir,
    save_path="indian_games_detector.pth",
    max_epochs=15,
    patience=3,
    batch_size=32,
    lr=0.001
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("   INDIAN TRADITIONAL GAMES CUSTOM DETECTOR TRAINING")
    print("=" * 70)
    print(f"Device               : {device}")
    print(f"Annotations Directory: {annotations_dir}")
    print(f"Images Directory     : {images_dir}")
    print(f"Target Checkpoint    : {save_path}")
    print(f"Max Epochs           : {max_epochs}")
    print(f"Early Stopping Ptc.  : {patience} consecutive epochs")
    print(f"Batch Size           : {batch_size}")
    print(f"Learning Rate        : {lr}")
    print("=" * 70)

    # 1. Dataset & DataLoader setup
    dataset = IndianGamesDatasetLoader(annotations_dir=annotations_dir, images_dir=images_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Zero-worker mode for cross-platform stability
        pin_memory=(device.type == "cuda")
    )
    print(f"[DATASET] Loaded {len(dataset)} augmented samples ({len(dataloader)} batches/epoch).\n")

    # 2. Model Initialization
    model = IndianTraditionalGameDetector(num_classes=6).to(device)

    # 3. Loss Functions & Optimizer Setup
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Early Stopping tracking variables
    best_loss = float("inf")
    epochs_no_improve = 0
    training_start_time = time.time()

    print("[TRAINING] Commencing training loop...\n")

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        
        running_total_loss = 0.0
        running_cls_loss = 0.0
        running_reg_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for batch_idx, (images, targets) in enumerate(dataloader):
            labels, bboxes = targets
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()

            cls_logits, bbox_coords = model(images)

            # Combined loss calculation: CrossEntropyLoss + SmoothL1Loss
            loss_cls = cls_criterion(cls_logits, labels)
            loss_reg = reg_criterion(bbox_coords, bboxes)
            loss = loss_cls + loss_reg

            loss.backward()
            optimizer.step()

            # Metric tracking
            batch_size_curr = images.size(0)
            running_total_loss += loss.item() * batch_size_curr
            running_cls_loss += loss_cls.item() * batch_size_curr
            running_reg_loss += loss_reg.item() * batch_size_curr
            
            _, predicted = torch.max(cls_logits, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += batch_size_curr

            if batch_idx % 10 == 0:
                elapsed_time = time.time() - epoch_start_time
                print(f"Epoch [{epoch}/{max_epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Elapsed Time: {elapsed_time:.1f}s | Loss: {loss.item():.4f}", flush=True)

        # Epoch statistics
        epoch_time = time.time() - epoch_start_time
        epoch_loss = running_total_loss / total_samples
        epoch_cls_loss = running_cls_loss / total_samples
        epoch_reg_loss = running_reg_loss / total_samples
        epoch_acc = (correct_preds / total_samples) * 100.0

        # Required print statement for epoch loss and time elapsed
        print(f"Epoch [{epoch}/{max_epochs}] Summary - Loss: {epoch_loss:.4f} (Cls: {epoch_cls_loss:.4f}, Reg: {epoch_reg_loss:.4f}) | Acc: {epoch_acc:5.2f}% | Time Elapsed: {epoch_time:.2f}s", flush=True)

        # Early Stopping Mechanism
        # Monitor total loss: if loss fails to decrease for 3 consecutive epochs, break
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  -> [EarlyStopping] Total loss did not improve ({epoch_loss:.4f} >= {best_loss:.4f}). Counter: {epochs_no_improve}/{patience}", flush=True)
            if epochs_no_improve >= patience:
                print(f"\n[EARLY STOPPING TRIGGERED] Total loss failed to decrease for {patience} consecutive epochs.", flush=True)
                print(f"Terminating training early at Epoch {epoch}.", flush=True)
                break

    total_training_duration = time.time() - training_start_time
    print("\n" + "=" * 70)
    print(f"[COMPLETED] Training concluded in {total_training_duration:.2f}s.")
    print(f"Best Total Loss Achieved: {best_loss:.4f}")

    # ==============================================================================
    # Component 4: Model Saving
    # Save the final trained model's state dictionary exactly as 'indian_games_detector.pth'
    # ==============================================================================
    save_file_path = Path(save_path).resolve()
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_file_path))
    print(f"[PERSISTENCE] State dictionary saved exactly as: {save_file_path.name}")
    print(f"Full Checkpoint Path: {save_file_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train Custom Indian Traditional Game Detector from Scratch")
    parser.add_argument("--annotations-dir", type=str, default=r"E:\IndianToysAndGames\IndianGamesDataset\Augmented\Annotations", help="Path to XML annotations")
    parser.add_argument("--images-dir", type=str, default=r"E:\IndianToysAndGames\IndianGamesDataset\Augmented\JPEGImages", help="Path to JPEG images")
    parser.add_argument("--save-path", type=str, default="indian_games_detector.pth", help="Target checkpoint filename")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (consecutive epochs)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")

    args = parser.parse_args()

    train_model(
        annotations_dir=args.annotations_dir,
        images_dir=args.images_dir,
        save_path=args.save_path,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == "__main__":
    main()
