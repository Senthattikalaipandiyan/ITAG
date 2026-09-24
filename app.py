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
    # Map to CPU in case the web server doesn't have a GPU
    device = torch.device("cpu")
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    state_dict = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Class map matching trained dataset labels
class_map = {
    0: "Background",
    1: "Spinning Top",
    2: "Seven Stones",
    3: "Thattangal",
    4: "Thayam",
    5: "Snakes and Ladders"
}
CLASS_MAP = class_map


# ==============================================================================
# 2. Build the Web App UI
# ==============================================================================
st.title("🎯 Indian Traditional Games Detector")
st.write("Upload a raw image to test the custom Single-Shot CNN.")

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
        if model is not None:
            with torch.no_grad():
                class_logits, bbox_coords = model(input_tensor)
                
            # Calculate probabilities and extract the winning class
            probabilities = torch.softmax(class_logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()

            # 5. Draw the Bounding Box and Label
            if predicted_class != 0 and confidence > confidence_threshold:
                # Convert PIL image to OpenCV BGR format for drawing
                img_cv = np.array(image)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                height, width, _ = img_cv.shape
                
                # Un-scale the [0, 1] bounding box back to actual pixel dimensions
                box = bbox_coords[0].numpy()
                x1, y1 = int(box[0] * width), int(box[1] * height)
                x2, y2 = int(box[2] * width), int(box[3] * height)
                
                # Dynamic drawing thickness based on image dimensions
                box_thick = max(2, int(min(width, height) / 250))
                font_scale = max(0.7, min(width, height) / 750)
                font_thick = max(1, int(font_scale * 2))

                # Draw the red rectangle and text
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), box_thick)
                label = f"{class_map[predicted_class]} ({confidence*100:.1f}%)"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                text_y = max(th + 10, y1 - 15)
                text_x = max(5, x1)

                # Filled background banner for label readability
                cv2.rectangle(
                    img_cv,
                    (text_x - 3, text_y - th - 5),
                    (text_x + tw + 3, text_y + baseline + 3),
                    (0, 0, 255),
                    cv2.FILLED
                )
                cv2.putText(img_cv, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
                
                # Convert back to RGB and display the final result in the browser
                result_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                st.success("Detection Complete!")
                st.image(result_img, caption="Inference Result", use_container_width=True)
                
                # --- NEW INNOVATION: GAME RULES & VIDEO TUTORIAL ---
                st.markdown("---")
                st.header(f"How to Play: {class_map[predicted_class]}")
                
                # Dictionary storing the rules and tutorial links for each game
                game_info = {
                    "Thayam": {
                        "rules": """
* Players roll wooden or brass dice (Dhaayakattai) to move their pieces in a clockwise direction along the board.
* A player must roll a "Dhaayam" (zero on one die, one on the other) to move a piece out of their home base.
* Rolling a one, six, twelve, or a Dhaayam grants the player an additional roll.
* Pieces can "cut" an opponent's piece by landing on the same square, sending the opponent's piece back to the start.
* Pieces cannot be cut if they are resting in marked safe zones.
""",
                        "video": "https://www.youtube.com/watch?v=YOUR_YOUTUBE_LINK_HERE" 
                    },
                    "Spinning Top": {
                        "rules": """
* Players wind a string tightly around a wooden top equipped with a pointed center nail.
* During the initial toss, players simultaneously throw and unwind their tops onto the ground.
* Players must quickly use their string to catch and lift the spinning top off the ground.
* If a player's top fails to spin on its nail ("Mattai") or they fail to catch it with the string, they lose the toss.
* Players who lose the toss place their tops in a center circle, while the winners try to strike them with their spinning tops to leave a dent ("Aakkar").
""",
                        "video": "https://www.youtube.com/watch?v=YOUR_YOUTUBE_LINK_HERE"
                    },
                    "Seven Stones": {
                        "rules": """
* Two teams play with a ball and a central pile of seven stackable stones.
* The starting team stands roughly 20 feet away and gets three attempts to knock over the stone pile with the ball.
* Once the pile is knocked over, the opposing team retrieves the ball and attempts to hit members of the first team below the knees to eliminate them.
* The first team attempts to restack the pile and trace a circle around it three times before all their players are eliminated.
* The defending team cannot move while holding the ball and must pass or throw it within 60 seconds.
""",
                        "video": "https://www.youtube.com/watch?v=YOUR_YOUTUBE_LINK_HERE"
                    },
                    "Snakes and Ladders": {
                        "rules": """
* Players navigate a 100-square grid by rolling dice.
* A player must roll a one to place their game piece onto the starting board.
* Rolling a one, five, or six grants an extra turn, but rolling three sixes in a row sends the player back to the start.
* Landing at the bottom of a ladder allows a player to advance to the top, representing virtues.
* Landing on the head of a snake forces the player down to its tail, representing vices or sins.
* The first player to reach the 100th square wins.
""",
                        "video": "https://www.youtube.com/watch?v=YOUR_YOUTUBE_LINK_HERE"
                    },
                    "Thattangal": {
                        "rules": """
* Players toss a single main stone into the air with one hand.
* While the main stone is airborne, the player must quickly scoop up a specific number of stones resting on the ground.
* The player must then catch the falling main stone in the same hand before it touches the ground.
* Players progress through increasingly difficult rounds, picking up stones in varying combinations and patterns.
* Failing to catch the tossed stone or dropping the gathered stones results in a loss of turn.
""",
                        "video": "https://www.youtube.com/watch?v=YOUR_YOUTUBE_LINK_HERE"
                    }
                }
                
                # Retrieve the specific game info based on the neural network's prediction
                info = game_info.get(class_map[predicted_class])
                
                if info:
                    # Use st.markdown to render the bullet points correctly
                    st.markdown(info["rules"])
                    # Streamlit automatically embeds the YouTube video 
                    st.video(info["video"])
            else:
                st.warning(f"No objects detected. (Highest match: {class_map.get(predicted_class, 'Unknown')} at {confidence*100:.1f}%)")

            # Expandable probability breakdown
            with st.expander("📊 View Detailed Class Probabilities"):
                prob_list = probabilities[0].tolist()
                for c_idx, c_prob in enumerate(prob_list):
                    st.progress(c_prob, text=f"{class_map.get(c_idx, 'Class ' + str(c_idx))}: {c_prob * 100:.2f}%")
