# 🎯 ITAG: Indian Traditional Games Detector

An end-to-end Computer Vision and Deep Learning application built with PyTorch, OpenCV, and Streamlit to detect and localize Indian traditional games in images.

---

## 🎲 Recognized Games
1. **Thayam** (Ancient Tamil dice board game)
2. **Seven Stones / Lagori**
3. **Spinning Top / Pambaram**
4. **Snakes & Ladders / Paramapadham**
5. **Thattangal** (Five Stones)

---

## 🧠 Model Architecture
- **Backbone**: Custom 4-Stage Convolutional Neural Network built from scratch with PyTorch.
- **Classification Head**: 6-class softmax output layer.
- **Bounding Box Regressor**: 4-coordinate normalized `[0, 1]` spatial regressor with Sigmoid activation.
- **Trained Weights**: `indian_games_detector.pth`

---

## 🚀 Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Senthattikalaipandiyan/ITAG.git
   cd ITAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deploying to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Connect your GitHub account and select this repository: `Senthattikalaipandiyan/ITAG`.
3. Set the Main file path to `app.py`.
4. Click **Deploy!**
