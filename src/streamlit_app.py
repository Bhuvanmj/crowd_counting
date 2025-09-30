import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# CSRNet Model
class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        vgg = models.vgg16_bn(pretrained=load_weights)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

# Streamlit settings
st.set_page_config(page_title="Crowd Counting Dashboard", layout="wide")
st.title("Crowd Counting Dashboard")
threshold = st.sidebar.slider("Crowd Threshold", 10, 500, 50, 5)
frame_skip = st.sidebar.number_input("Frame Interval (process every Nth frame)", min_value=1, value=5)
video_file = st.file_uploader("Upload Video (mp4)", type=["mp4"])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./best_csrnet.pth"
model = CSRNet(load_weights=False).to(device)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# Streamlit containers
frame_display = st.empty()
count_display = st.empty()
alert_display = st.empty()

# Video processing
if video_file:
    tfile = "./temp_video.mp4"
    with open(tfile, 'wb') as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(tfile)
    frame_count = 0
    crowd_counts = []
    peak_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_t)
            output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
            density_map = output[0,0].cpu().numpy()
            count = density_map.sum()

        crowd_counts.append(count)
        peak_count = max(peak_count, count)
        avg_count = np.mean(crowd_counts)

        fig, axes = plt.subplots(1,2, figsize=(12,5))
        axes[0].imshow(img_pil)
        axes[0].axis('off')
        axes[0].set_title("Original Frame")
        axes[1].imshow(density_map, cmap='hot')
        axes[1].axis('off')
        axes[1].set_title(f"Predicted Count = {count:.2f}")

        frame_display.pyplot(fig)
        count_display.metric("Current Crowd Count", f"{count:.2f}", delta=f"Peak: {peak_count:.2f} | Avg: {avg_count:.2f}")

        if count >= threshold:
            alert_display.warning(f"Overcrowding Detected! Count = {count:.2f}")
        else:
            alert_display.success("Crowd below threshold")

        time.sleep(0.1)

    cap.release()
    st.success("Video processing completed!")
