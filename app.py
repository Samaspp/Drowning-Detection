import streamlit as st
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image
import tempfile
import albumentations

# Load Label Binarizer
lb = joblib.load("lb.pkl")

# Define Custom CNN Model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, len(lb.classes_))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Image Preprocessing
transform = albumentations.Compose([albumentations.Resize(224, 224)])

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, "Error: Cannot open video file."

    frame_width, frame_height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (frame_width, frame_height))

    frame_display = st.empty()  # Streamlit real-time frame display
    alert_placeholder = st.sidebar.empty()  # Alert message on the right side
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bbox, label, conf = cv.detect_common_objects(frame)
        isDrowning = False

        for i, lbl in enumerate(label):
            if lbl == "person":
                x1, y1, x2, y2 = bbox[i]
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size > 0:
                    with torch.no_grad():
                        pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                        pil_image = transform(image=np.array(pil_image))["image"]
                        pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                        pil_image = torch.tensor(pil_image, dtype=torch.float).to(device).unsqueeze(0)
                        outputs = model(pil_image)
                        _, preds = torch.max(outputs.data, 1)
                    
                    predicted_label = lb.classes_[preds.item()]
                    if predicted_label == "drowning":
                        isDrowning = True
                        alert_placeholder.warning("🚨 ALERT: Drowning Detected! 🚨")
                    else:
                        alert_placeholder.empty()
        
        # Draw bounding boxes for all detections
        out_frame = draw_bbox(frame, bbox, label, conf, isDrowning)
        out.write(out_frame)
        frame_display.image(out_frame, channels="BGR")  # Update Streamlit frame display

    cap.release()
    out.release()
    return temp_output.name, isDrowning

# Streamlit UI
st.set_page_config(page_title="Drowning Detection", layout="wide")

st.title("Drowning Detection System")
st.write("Upload a video, and the system will detect drowning incidents in real-time.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing video..."):
        output_video_path, detected_drowning = process_video(temp_video_path)

    if output_video_path:
        st.video(output_video_path)
        st.success("Video processed successfully.")
    else:
        st.error("Error processing video.")
