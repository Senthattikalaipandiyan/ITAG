"""
11_static_inference.py
======================
Static Object Detection and Bounding Box Inference for Hackathon Presentation.

Pipeline:
1. Loads custom PyTorch architecture `IndianTraditionalGameDetector`.
2. Loads trained weights from `indian_games_detector.pth`.
3. Processes specified dataset images through OpenCV and `torchvision.transforms`:
   - BGR -> RGB conversion
   - ToPILImage() -> Resize((512, 512)) -> ToTensor() -> .unsqueeze(0)
4. Predicts game class and [0, 1] normalized bounding box coordinates.
5. Projects bounding box back to original high-resolution image coordinates.
6. Annotates image with thick red bounding box and predicted game name.
7. Saves outputs as `demo_output_1.jpg`, `demo_output_2.jpg`, etc., in project root.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Base paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PTH = BASE_DIR / "indian_games_detector.pth"
WEIGHTS_PKL = BASE_DIR / "indian_games_detector.pkl"
DATASET_JPEG_DIR = BASE_DIR / "IndianGamesDataset" / "JPEGImages"

# Class label mapping matching dataset annotations and training parser
CLASS_NAMES = {
    0: "Background",
    1: "SpinningTop",
    2: "SevenStones",
    3: "Thattangal",
    4: "Thayam",
    5: "SnakesandLadders"
}


# ==============================================================================
# Model Architecture: IndianTraditionalGameDetector
# ==============================================================================
class IndianTraditionalGameDetector(nn.Module):
    """
    Exact dual-output architecture matching trained weights:
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
            nn.Sigmoid()  # Restricts predictions to [0, 1]
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
# Helper: Load Model
# ==============================================================================
def load_trained_model(device):
    """Initializes model and loads weights from .pth or .pkl checkpoint."""
    model = IndianTraditionalGameDetector(num_classes=6).to(device)
    
    if WEIGHTS_PTH.exists():
        print(f"[MODEL] Loading weights from: {WEIGHTS_PTH.name}...")
        state_dict = torch.load(str(WEIGHTS_PTH), map_location=device)
    elif WEIGHTS_PKL.exists():
        import pickle
        print(f"[MODEL] Loading weights from: {WEIGHTS_PKL.name}...")
        with open(WEIGHTS_PKL, "rb") as f:
            state_dict = pickle.load(f)
    else:
        raise FileNotFoundError("Could not find 'indian_games_detector.pth' or 'indian_games_detector.pkl'!")

    model.load_state_dict(state_dict)
    model.eval()
    print("[MODEL] Model initialized and set to eval() mode.\n")
    return model


# ==============================================================================
# Helper: Resolve Image Path
# ==============================================================================
def resolve_image_path(img_name_or_path):
    """Finds image in current dir, JPEGImages dataset dir, or absolute path."""
    p = Path(img_name_or_path)
    if p.is_file():
        return p
    # Check dataset folder
    dataset_p = DATASET_JPEG_DIR / p.name
    if dataset_p.is_file():
        return dataset_p
    # Check project root
    root_p = BASE_DIR / p.name
    if root_p.is_file():
        return root_p
    return None


# ==============================================================================
# Main Static Inference Routine
# ==============================================================================
def run_static_inference():
    print("=" * 70)
    print("   INDIAN TRADITIONAL GAMES - STATIC INFERENCE DEMO GENERATOR")
    print("=" * 70)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device : {device}")

    # 2. Load trained model
    model = load_trained_model(device)

    # 3. Target Files (User customizable list of test images)
    test_images = [
        "IMG_20260923_164828967_HDR.jpg",
        "IMG_20260923_165157372_HDR.jpg",
        "IMG_20260923_164922053.jpg"
    ]

    # 4. Preprocessing transform pipeline
    preprocess_pipeline = transforms.Compose([
        transforms.ToPILImage(),        # Converts RGB NumPy array to PIL Image
        transforms.Resize((512, 512)),  # Resizes to model training dimension
        transforms.ToTensor()           # Scales 0-255 pixels down to [0.0, 1.0] float tensor
    ])

    saved_outputs = []

    # 5. Inference and Drawing Loop
    for idx, img_ref in enumerate(test_images, start=1):
        img_path = resolve_image_path(img_ref)
        if img_path is None:
            print(f"[WARNING] Skipping image '{img_ref}' - file not found.")
            continue

        print(f"[{idx}/{len(test_images)}] Processing: {img_path.name}")

        # Load original high-resolution image with OpenCV
        orig_image = cv2.imread(str(img_path))
        if orig_image is None:
            print(f"[ERROR] Failed to decode image: {img_path}")
            continue

        orig_h, orig_w = orig_image.shape[:2]

        # Convert BGR to RGB for PyTorch transforms
        rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        # Apply preprocessing transforms and add batch dimension
        input_tensor = preprocess_pipeline(rgb_image).unsqueeze(0).to(device)

        # Model Inference
        with torch.no_grad():
            cls_logits, bbox_coords = model(input_tensor)
            probabilities = torch.softmax(cls_logits, dim=1)
            confidence, pred_class_idx = torch.max(probabilities, dim=1)

        conf_val = confidence.item()
        pred_idx = pred_class_idx.item()
        pred_name = CLASS_NAMES.get(pred_idx, "Unknown")
        bbox_norm = bbox_coords[0].cpu().numpy()

        xmin_norm, ymin_norm, xmax_norm, ymax_norm = bbox_norm

        # Map [0, 1] scaled bounding box coordinates back to original high-res dimensions
        x1 = max(0, min(orig_w - 1, int(xmin_norm * orig_w)))
        y1 = max(0, min(orig_h - 1, int(ymin_norm * orig_h)))
        x2 = max(0, min(orig_w - 1, int(xmax_norm * orig_w)))
        y2 = max(0, min(orig_h - 1, int(ymax_norm * orig_h)))

        # Ensure valid coordinate order
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # Calculate dynamic line thickness and font scale proportional to high-res image
        box_thickness = max(4, int(min(orig_w, orig_h) / 200))
        font_scale = max(1.0, min(orig_w, orig_h) / 750)
        font_thickness = max(2, int(font_scale * 2.2))

        # Annotate: Draw thick red bounding box: (0, 0, 255)
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 0, 255), thickness=box_thickness)

        # Formulate label with class name and confidence percentage
        label_text = f"{pred_name} ({conf_val * 100:.1f}%)"

        # Calculate text background box for crystal-clear readability
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        text_y = max(text_h + 15, y1 - 10)
        text_x = max(10, x1)

        # Draw red label background banner
        cv2.rectangle(
            orig_image,
            (text_x - 5, text_y - text_h - 10),
            (text_x + text_w + 5, text_y + baseline + 5),
            (0, 0, 255),
            cv2.FILLED
        )

        # Write white text over red banner
        cv2.putText(
            orig_image,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

        # Save output image to project root
        output_filename = f"demo_output_{idx}.jpg"
        output_path = BASE_DIR / output_filename
        success = cv2.imwrite(str(output_path), orig_image)

        if success:
            saved_outputs.append((output_path, pred_name, conf_val, (x1, y1, x2, y2)))
            print(f"      -> Predicted : {pred_name} ({conf_val * 100:.2f}%)")
            print(f"      -> BBox [px] : ({x1}, {y1}) to ({x2}, {y2})")
            print(f"      -> Saved to  : {output_filename}\n")
        else:
            print(f"[ERROR] Failed to save {output_filename}")

    # Final summary banner
    print("=" * 70)
    print("   DEMO OUTPUTS GENERATED SUCCESSFULLY FOR HACKATHON")
    print("=" * 70)
    for out_path, pred_name, conf, box in saved_outputs:
        print(f" - {out_path.name:18} | Class: {pred_name:15} | Conf: {conf*100:6.2f}% | Box: {box}")
    print("\nYou can now open demo_output_1.jpg, demo_output_2.jpg, etc., for your presentation!\n")


if __name__ == "__main__":
    run_static_inference()
