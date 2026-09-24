"""
5_evaluate_model.py
===================
Model Evaluation and Inference Script for Indian Traditional Game Detector.

Components:
1. Load Model: Initializes exact IndianTraditionalGameDetector architecture and loads 'indian_games_detector.pth'.
2. Load Test Data: Uses a custom Dataset/DataLoader to evaluate original un-augmented test images and XMLs.
3. Calculate Metrics: Evaluates under torch.no_grad() and computes:
   - Overall Classification Accuracy (%)
   - Average Intersection over Union (IoU) for bounding boxes
4. Visual Proof: Selects 3 random test images, draws predicted bounding boxes (in red) and class labels,
   and saves them as test_result_1.jpg, test_result_2.jpg, and test_result_3.jpg in the main folder.
"""

import os
import sys
import random
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
# Component 1: Architecture Definition (IndianTraditionalGameDetector)
# ==============================================================================
class IndianTraditionalGameDetector(nn.Module):
    """
    Exact dual-output architecture matching 4_train_model.py:
    - 4-Block Conv2d Feature Extractor
    - 6-Class Linear Classifier Head
    - 4-Coordinate Linear Regressor Head with Sigmoid ([0, 1] normalized bounding box)
    """
    def __init__(self, num_classes=6):
        super(IndianTraditionalGameDetector, self).__init__()
        
        # Block 1: (3, 512, 512) -> (32, 128, 128)
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
        
        # Spatial pooling to 1,024 features
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        flattened_dim = 256 * 2 * 2  # 1024
        
        # 6-Class Linear Classifier Head
        self.classifier_head = nn.Linear(flattened_dim, num_classes)
        
        # 4-Coordinate Linear Regressor Head with Sigmoid Activation
        self.regressor_head = nn.Sequential(
            nn.Linear(flattened_dim, 4),
            nn.Sigmoid()
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
# Component 2: Test Dataset Loader
# ==============================================================================
class IndianGamesTestDataset(Dataset):
    """
    Dataset loader for original un-augmented test split samples.
    """
    def __init__(self, split_txt_path, images_dir, annotations_dir, target_size=(512, 512)):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.target_size = target_size
        
        with open(split_txt_path, "r") as f:
            self.sample_ids = [line.strip().split()[0] for line in f if line.strip()]
            
        if len(self.sample_ids) == 0:
            raise RuntimeError(f"No sample IDs found in {split_txt_path}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_path = self.images_dir / f"{sample_id}.jpg"
        xml_path = self.annotations_dir / f"{sample_id}.xml"
        
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read test image: {img_path}")
            
        orig_h, orig_w = image.shape[:2]
        
        # Parse ground-truth from XML
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
        
        size_elem = root.find("size")
        if size_elem is not None:
            w_tag = size_elem.find("width")
            h_tag = size_elem.find("height")
            if w_tag is not None and h_tag is not None:
                orig_w = float(w_tag.text)
                orig_h = float(h_tag.text)
                
        objs = root.findall("object")
        if len(objs) == 0:
            gt_label = CLASS_TO_IDX["background"]
            gt_bbox = [0.0, 0.0, 1.0, 1.0]
        else:
            primary_name = objs[0].find("name").text.strip()
            gt_label = CLASS_TO_IDX.get(primary_name, CLASS_TO_IDX["background"])
            
            xmins, ymins, xmaxs, ymaxs = [], [], [], []
            for obj in objs:
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmins.append(float(bndbox.find("xmin").text))
                    ymins.append(float(bndbox.find("ymin").text))
                    xmaxs.append(float(bndbox.find("xmax").text))
                    ymaxs.append(float(bndbox.find("ymax").text))
                    
            if len(xmins) > 0:
                gt_bbox = [
                    max(0.0, min(1.0, min(xmins) / orig_w)),
                    max(0.0, min(1.0, min(ymins) / orig_h)),
                    max(0.0, min(1.0, max(xmaxs) / orig_w)),
                    max(0.0, min(1.0, max(ymaxs) / orig_h))
                ]
            else:
                gt_bbox = [0.0, 0.0, 1.0, 1.0]
                
        # Resize and normalize for model input
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        
        return {
            "image": img_tensor,
            "gt_label": torch.tensor(gt_label, dtype=torch.long),
            "gt_bbox": torch.tensor(gt_bbox, dtype=torch.float32),
            "sample_id": sample_id,
            "orig_path": str(img_path),
            "orig_size": (orig_w, orig_h)
        }


# ==============================================================================
# Helper: Intersection over Union (IoU) Calculation
# ==============================================================================
def calculate_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two bounding boxes
    in format [xmin, ymin, xmax, ymax].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


# ==============================================================================
# Main Evaluation Routine
# ==============================================================================
def evaluate(
    weights_path="indian_games_detector.pth",
    dataset_dir="IndianGamesDataset",
    output_dir="."
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / dataset_dir
    weights_file = base_dir / weights_path
    out_path = base_dir / output_dir

    print("=" * 70)
    print("   INDIAN TRADITIONAL GAMES DETECTOR - TEST EVALUATION")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Model Weights : {weights_file}")
    print(f"Dataset Path  : {dataset_path}")
    print("=" * 70)

    # 1. Load Model Architecture & Weights
    if not weights_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_file}")

    model = IndianTraditionalGameDetector(num_classes=6).to(device)
    state_dict = torch.load(str(weights_file), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("[1/4] Model loaded successfully and set to eval() mode.\n")

    # 2. Determine test split file
    test_txt = dataset_path / "ImageSets" / "Main" / "test.txt"
    if not test_txt.exists():
        # Fallback to val.txt if test.txt is absent
        test_txt = dataset_path / "ImageSets" / "Main" / "val.txt"
        
    if not test_txt.exists():
        raise FileNotFoundError(f"Test split file not found in {dataset_path / 'ImageSets' / 'Main'}")

    images_dir = dataset_path / "JPEGImages"
    annotations_dir = dataset_path / "Annotations"

    test_dataset = IndianGamesTestDataset(
        split_txt_path=test_txt,
        images_dir=images_dir,
        annotations_dir=annotations_dir
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"[2/4] Loaded {len(test_dataset)} unseen test images from: {test_txt.name}\n")

    # 3. Calculate Evaluation Metrics
    print("[3/4] Running inference over test set...")
    total_samples = 0
    correct_classifications = 0
    all_ious = []
    
    per_class_total = {k: 0 for k in CLASS_TO_IDX.values()}
    per_class_correct = {k: 0 for k in CLASS_TO_IDX.values()}
    per_class_ious = {k: [] for k in CLASS_TO_IDX.values()}

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            gt_label = batch["gt_label"].item()
            gt_bbox = batch["gt_bbox"][0].numpy()

            cls_logits, bbox_coords = model(image)

            pred_class = torch.argmax(cls_logits, dim=1).item()
            pred_bbox = bbox_coords[0].cpu().numpy()

            is_correct = (pred_class == gt_label)
            if is_correct:
                correct_classifications += 1
                per_class_correct[gt_label] += 1

            iou = calculate_iou(pred_bbox, gt_bbox)
            all_ious.append(iou)

            per_class_total[gt_label] += 1
            per_class_ious[gt_label].append(iou)
            total_samples += 1

    overall_accuracy = (correct_classifications / total_samples) * 100.0
    mean_iou = sum(all_ious) / len(all_ious) if len(all_ious) > 0 else 0.0

    print("\n" + "=" * 70)
    print("                    EVALUATION METRICS RESULTS")
    print("=" * 70)
    print(f"Total Test Images Evaluated : {total_samples}")
    print(f"Classification Accuracy     : {overall_accuracy:.2f}% ({correct_classifications}/{total_samples})")
    print(f"Average Bounding Box IoU    : {mean_iou:.4f} (mIoU)")
    print("-" * 70)
    print("Per-Class Breakdown:")
    for cls_name, cls_idx in CLASS_TO_IDX.items():
        count = per_class_total[cls_idx]
        if count > 0:
            c_acc = (per_class_correct[cls_idx] / count) * 100.0
            c_iou = sum(per_class_ious[cls_idx]) / count
            print(f"  - {cls_name:18s} | Count: {count:2d} | Acc: {c_acc:6.2f}% | Mean IoU: {c_iou:.4f}")
    print("=" * 70 + "\n")

    # 4. Visual Proof: Pick 3 random images, run inference, draw predicted red box & label
    print("[4/4] Generating visual proof images...")
    random.seed(42)  # Fixed seed for reproducible sample selection
    selected_indices = random.sample(range(len(test_dataset)), 3)

    for i, idx in enumerate(selected_indices, 1):
        sample = test_dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        sample_id = sample["sample_id"]
        orig_img_path = sample["orig_path"]

        # Run inference
        with torch.no_grad():
            cls_logits, bbox_coords = model(image_tensor)
            probs = torch.softmax(cls_logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
            pred_bbox = bbox_coords[0].cpu().numpy()

        pred_class_name = IDX_TO_CLASS.get(pred_class, f"Class_{pred_class}")

        # Read original full-resolution image for high-quality visualization
        orig_image = cv2.imread(orig_img_path)
        h, w = orig_image.shape[:2]

        # Convert normalized [0, 1] predicted coordinates to pixel coordinates
        xmin = int(np.clip(pred_bbox[0] * w, 0, w - 1))
        ymin = int(np.clip(pred_bbox[1] * h, 0, h - 1))
        xmax = int(np.clip(pred_bbox[2] * w, 0, w - 1))
        ymax = int(np.clip(pred_bbox[3] * h, 0, h - 1))

        # 1. Draw predicted bounding box in RED: (0, 0, 255) in BGR
        box_thickness = max(3, int(min(h, w) * 0.003))
        cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=box_thickness)

        # 2. Draw predicted class label text banner above the bounding box
        label_text = f"{pred_class_name} ({confidence * 100:.1f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, min(h, w) * 0.0008)
        font_thickness = max(2, int(font_scale * 2.5))
        
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        text_w, text_h = text_size[0], text_size[1]
        
        # Background rectangle for crisp text visibility
        text_bg_y1 = max(0, ymin - text_h - 14)
        text_bg_y2 = ymin
        text_bg_x2 = min(w, xmin + text_w + 14)
        cv2.rectangle(orig_image, (xmin, text_bg_y1), (text_bg_x2, text_bg_y2), (0, 0, 255), -1)
        
        # White text inside the red banner
        text_x = xmin + 7
        text_y = ymin - 7
        cv2.putText(orig_image, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Save output image
        out_filename = f"test_result_{i}.jpg"
        save_dest = out_path / out_filename
        cv2.imwrite(str(save_dest), orig_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  -> Saved {out_filename} ({sample_id}): Predicted '{pred_class_name}' with bbox [{xmin}, {ymin}, {xmax}, {ymax}]")

    print("\n[SUCCESS] Visual verification images generated successfully in main workspace folder.")


if __name__ == "__main__":
    evaluate()
