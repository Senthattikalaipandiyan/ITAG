import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

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


@st.cache_resource
def load_model():
    model = IndianTraditionalGameDetector(num_classes=6)
    # Map to CPU in case the web server doesn't have a GPU
    device = torch.device('cpu')
    model_file = str(MODEL_PATH) if MODEL_PATH.exists() else "indian_games_detector.pth"
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model


# 2. Build the Web App UI
st.title("Indian Traditional Games Detector")
st.write("Upload a raw image to test the custom Single-Shot CNN.")

# Create the upload widget
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the original upload
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Upload", use_container_width=True)
    
    with st.spinner("Running PyTorch Inference..."):
        # 3. Preprocess the image exactly like your dataloader
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        # 4. Pass the tensor through the dual-head network
        model = load_model()
        with torch.no_grad():
            class_logits, bbox_coords = model(input_tensor)
            
        # Calculate probabilities and extract the winning class
        probabilities = torch.softmax(class_logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()

        # Class dictionary mapped exactly to the dataset XML annotations
        class_map = {
            0: "Background",
            1: "Spinning Top",
            2: "Seven Stones",
            3: "Thattangal",
            4: "Thayam",
            5: "Snakes and Ladders"
        }

        # 5. Draw the Bounding Box and Label
        if predicted_class != 0 and confidence > 0.60:
            # Convert PIL image to OpenCV BGR format for drawing
            img_cv = np.array(image)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            height, width, _ = img_cv.shape
            
            # Un-scale the [0, 1] bounding box back to actual pixel dimensions
            box = bbox_coords[0].numpy()
            x1, y1 = int(box[0] * width), int(box[1] * height)
            x2, y2 = int(box[2] * width), int(box[3] * height)
            
            # Draw the red rectangle and text
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"{class_map[predicted_class]} ({confidence*100:.1f}%)"
            cv2.putText(img_cv, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert back to RGB and display the final result in the browser
            result_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            st.success("Detection Complete!")
            st.image(result_img, caption="Inference Result", use_container_width=True)

            # --- GAME RULES & VIDEO TUTORIAL ---
            st.markdown("---")
            st.header(f"How to Play: {class_map[predicted_class]}")

            game_tutorials = {
                "Thayam": {
                    "rules": """
* Roll the dice to enter coins onto the board.
* A roll of 1 (Dhaayam) is required to release a piece from the home base.
* Move pieces along the outer track toward the inner victory square.
* Land on opponent pieces to eliminate them back to the start.
""",
                    "video": "https://www.youtube.com/watch?v=p_XEhnaYyIk"
                },
                "Spinning Top": {
                    "rules": """
* Wrap the cord tightly around the top starting from the peg.
* Throw the top forward while pulling the cord backward to initiate spin.
* Catch the spinning top on the palm or string to perform tricks.
""",
                    "video": "https://www.youtube.com/watch?v=sI_ca9lTvMU"
                },
                "Seven Stones": {
                    "rules": """
* Seekers attempt to knock down the stack of seven stones using a ball.
* Once knocked down, seekers must rebuild the stack while avoiding being hit by the ball.
* Hitters pass the ball among teammates to strike seekers before the stack is restored.
""",
                    "video": "https://www.youtube.com/watch?v=aZfE6nlFLkY"
                },
                "Snakes and Ladders": {
                    "rules": """
* Roll the die to advance across the numbered squares from 1 to 100.
* Ladders allow you to climb directly to higher squares.
* Landing on a snake's head forces you to slide down to its tail.
""",
                    "video": "https://www.youtube.com/watch?v=a-kTZF2EEKc"
                },
                "Thattangal": {
                    "rules": """
* Toss one stone into the air with one hand.
* Pick up target stones from the floor while the tossed stone is airborne.
* Catch the tossed stone before it hits the ground without dropping collected stones.
""",
                    "video": "https://www.youtube.com/watch?v=Mq7t4v5maeo"
                }
            }

            selected_game = game_tutorials.get(class_map[predicted_class])
            if selected_game:
                st.markdown(selected_game["rules"])
                st.video(selected_game["video"])

        else:
            st.warning(f"No objects detected. (Highest match: {class_map[predicted_class]} at {confidence*100:.1f}%)")
