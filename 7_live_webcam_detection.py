"""
7_live_webcam_detection.py
==========================
Real-Time Object Detection and Localization for Indian Traditional Games using Webcam.

Features:
- Loads custom IndianTraditionalGameDetector architecture without external backbones.
- Preprocesses webcam feed to (512, 512) float tensors via torchvision.transforms.
- Predicts game class and localized bounding boxes in real-time under torch.no_grad().
- Displays prominent red bounding boxes and class labels (suppressing Class 0: Background).
- Interactive OpenCV window with clean exit on pressing 'q'.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Base paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PTH = BASE_DIR / "indian_games_detector.pth"
WEIGHTS_PKL = BASE_DIR / "indian_games_detector.pkl"

# ==============================================================================
# Class Label Mapping (0 = Background, 1-5 = Indian Traditional Games)
# ==============================================================================
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
# Helper to Load Trained Weights
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
    print("[MODEL] Model loaded successfully and set to eval() mode.\n")
    return model


# ==============================================================================
# Main Webcam Live Detection Routine
# ==============================================================================
def run_live_webcam_detection():
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("   INDIAN TRADITIONAL GAMES - REAL-TIME WEBCAM DETECTOR")
    print("=" * 65)
    print(f"Device        : {device}")
    
    # 2. Load Model
    model = load_trained_model(device)

    # 3. Preprocessing Transform via torchvision.transforms
    preprocess_pipeline = transforms.Compose([
        transforms.ToPILImage(),        # Converts RGB NumPy array from cv2 to PIL Image
        transforms.Resize((512, 512)),  # Resizes to match training pipeline input dimensions
        transforms.ToTensor()           # Scales 0-255 uint8 pixel values down to 0.0-1.0 float tensor
    ])

    # 4. Initialize Webcam (DirectShow for Windows hardware reliability)
    print("[CAMERA] Connecting to webcam (index 0)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam at index 0. Check camera permissions or connection.")
        sys.exit(1)

    window_name = "Indian Games Custom Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[CAMERA] Live feed started. Press 'q' on the video window to quit.\n")

    # Real-time frame loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame from webcam. Exiting loop...")
            break

        orig_h, orig_w = frame.shape[:2]

        # 1. Convert raw BGR frame from cv2.VideoCapture to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Process via torchvision.transforms: ToPILImage() -> Resize((512, 512)) -> ToTensor()
        # 3. Add batch dimension with .unsqueeze(0) to get (1, 3, 512, 512)
        input_tensor = preprocess_pipeline(rgb_frame).unsqueeze(0).to(device)

        # Inference inside with torch.no_grad()
        with torch.no_grad():
            cls_logits, bbox_coords = model(input_tensor)
            softmax_output = torch.softmax(cls_logits, dim=1)
            
            # Diagnostic: Output raw probability scores for ALL 6 classes in real-time
            print(f"Raw Probs: {softmax_output[0].tolist()}", flush=True)

            max_prob, max_idx = torch.max(softmax_output, dim=1)
            confidence = max_prob.item()
            predicted_class = max_idx.item()
            pred_bbox = bbox_coords[0].cpu().numpy()

        # Visual Overlay: Strict filters to eliminate hallucinated boxes on empty walls
        # Must require BOTH: explicitly NOT background (class 0) AND confidence >= 0.92 (92%)
        if predicted_class != 0 and confidence >= 0.92:
            pred_name = CLASS_NAMES.get(predicted_class, f"Class {predicted_class}")
            
            # Map normalized [0, 1] coordinates back to original webcam frame dimensions
            xmin = int(np.clip(pred_bbox[0] * orig_w, 0, orig_w - 1))
            ymin = int(np.clip(pred_bbox[1] * orig_h, 0, orig_h - 1))
            xmax = int(np.clip(pred_bbox[2] * orig_w, 0, orig_w - 1))
            ymax = int(np.clip(pred_bbox[3] * orig_h, 0, orig_h - 1))

            # 1. Draw prominent RED bounding box: (0, 0, 255) in BGR
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)

            # 2. Write predicted game name above the box in clear, legible font
            label_text = f"{pred_name} ({confidence * 100:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            font_thickness = 2
            
            text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
            text_w, text_h = text_size[0], text_size[1]
            
            # Background rectangle for high-contrast text visibility
            banner_y1 = max(0, ymin - text_h - 12)
            banner_y2 = ymin
            banner_x2 = min(orig_w, xmin + text_w + 12)
            cv2.rectangle(frame, (xmin, banner_y1), (banner_x2, banner_y2), (0, 0, 255), -1)
            
            # White text on red banner
            cv2.putText(
                frame,
                label_text,
                (xmin + 6, ymin - 6),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )
        else:
            # Display search status when background or low confidence detection occurs
            status_text = "Status: Searching for Indian Traditional Games..."
            if predicted_class != 0 and confidence < 0.92:
                status_text = f"Status: Suppressed low-confidence detection ({confidence * 100:.1f}% < 92%)"

            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # Instructions banner at bottom
        cv2.putText(
            frame,
            "Press 'q' to Quit",
            (20, orig_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        # Display live feed
        cv2.imshow(window_name, frame)

        # Graceful exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[USER] 'q' pressed. Closing webcam feed...")
            break

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    print("[CLEANUP] Webcam released and all OpenCV windows closed successfully.")


if __name__ == "__main__":
    run_live_webcam_detection()
