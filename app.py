import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Indian Traditional Games Detector (ITAG)",
    page_icon="🎯",
    layout="centered"
)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "indian_games_detector.pth"

# ==============================================================================
# 1. Custom CNN Architecture Definition
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
# Model Loading (Cached)
# ==============================================================================
@st.cache_resource
def load_model():
    model = IndianTraditionalGameDetector(num_classes=6)
    # Map to CPU for cloud server compatibility
    device = torch.device("cpu")
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    state_dict = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Class map matching trained dataset labels
CLASS_MAP = {
    0: "Background",
    1: "Spinning Top",
    2: "Seven Stones",
    3: "Thattangal",
    4: "Thayam",
    5: "Snakes and Ladders"
}


# ==============================================================================
# 2. Build the Web App UI
# ==============================================================================
st.title("🎯 Indian Traditional Games Detector")
st.markdown(
    "Upload an image of an Indian traditional game to detect its identity and locate it "
    "using our custom **Single-Shot PyTorch Deep Learning Detector**."
)

with st.sidebar:
    st.header("About ITAG")
    st.info(
        "**Classes Recognized:**\n"
        "- 🎲 Thayam\n"
        "- 🪨 Seven Stones (Lagori)\n"
        "- 🌀 Spinning Top (Pambaram)\n"
        "- 🐍 Snakes & Ladders\n"
        "- 🖐️ Thattangal"
    )
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.30, max_value=0.99, value=0.60, step=0.05)

# Create the upload widget
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display the original upload
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    with st.spinner("Running PyTorch Inference..."):
        # 3. Preprocess the image exactly like training pipeline
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        # 4. Pass the tensor through the dual-head network
        model = load_model()
        if model is not None:
            with torch.no_grad():
                class_logits, bbox_coords = model(input_tensor)
                
            # Calculate probabilities and extract the winning class
            probabilities = torch.softmax(class_logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()

            with col2:
                st.subheader("Inference Result")
                # 5. Draw the Bounding Box and Label if valid detection
                if predicted_class != 0 and confidence >= confidence_threshold:
                    # Convert PIL image to OpenCV BGR format for drawing
                    img_cv = np.array(image)
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                    height, width, _ = img_cv.shape
                    
                    # Un-scale the [0, 1] bounding box back to actual pixel dimensions
                    box = bbox_coords[0].numpy()
                    x1 = max(0, min(width - 1, int(box[0] * width)))
                    y1 = max(0, min(height - 1, int(box[1] * height)))
                    x2 = max(0, min(width - 1, int(box[2] * width)))
                    y2 = max(0, min(height - 1, int(box[3] * height)))
                    
                    # Dynamic thickness based on image dimensions
                    box_thick = max(2, int(min(width, height) / 250))
                    font_scale = max(0.6, min(width, height) / 800)
                    font_thick = max(1, int(font_scale * 2))

                    # Draw red bounding rectangle
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), thickness=box_thick)
                    
                    label = f"{CLASS_MAP[predicted_class]} ({confidence*100:.1f}%)"
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                    text_y = max(th + 10, y1 - 10)
                    text_x = max(5, x1)

                    # Draw filled background banner for readability
                    cv2.rectangle(
                        img_cv,
                        (text_x - 3, text_y - th - 5),
                        (text_x + tw + 3, text_y + baseline + 3),
                        (0, 0, 255),
                        cv2.FILLED
                    )
                    cv2.putText(
                        img_cv, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA
                    )
                    
                    # Convert back to RGB and display
                    result_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    st.image(result_img, use_container_width=True)
                    st.success(f"**Detected:** {CLASS_MAP[predicted_class]} with **{confidence*100:.1f}%** confidence!")
                else:
                    st.image(image, use_container_width=True)
                    st.warning(f"No game detected above threshold. (Top guess: **{CLASS_MAP.get(predicted_class, 'None')}** at {confidence*100:.1f}%)")

            # Expandable probability breakdown
            with st.expander("📊 View Detailed Class Probabilities"):
                prob_list = probabilities[0].tolist()
                for c_idx, c_prob in enumerate(prob_list):
                    st.progress(c_prob, text=f"{CLASS_MAP.get(c_idx, 'Class ' + str(c_idx))}: {c_prob * 100:.2f}%")
