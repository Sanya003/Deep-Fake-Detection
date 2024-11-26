import streamlit as st
import torch
import torchvision.transforms as transforms
from torch import nn
import cv2
import numpy as np
import torch
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import face_recognition
import sys
import time
from torch.autograd import Variable
from function import Model, validation_dataset, predict

# Set Streamlit page config
st.set_page_config(
                    page_title="DeepFake Detection", 
                    layout="centered"
                )

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = Model(num_classes=2).cuda()
    model_path = "model.pt"  # Update with actual model path
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model()

st.title("DeepFake Detection App")
st.sidebar.header("Upload Video")
uploaded_video = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_video:
    st.sidebar.write("Video uploaded successfully!")

    # Save the uploaded video locally for processing
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())

    st.video("uploaded_video.mp4")

    # Preprocess and analyze the video
    st.write("Processing video...")

    try:
        # Validation dataset
        video_dataset = validation_dataset(
            video_names=["uploaded_video.mp4"], 
            sequence_length=20, 
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )

        video_frames = video_dataset[0]  # Extract frames from video
        if video_frames is not None:
            prediction = predict(model, video_frames)
            st.write(f"Prediction: {'REAL' if prediction[0] == 1 else 'FAKE'}")
            st.write(f"Accuracy: {prediction[1]:.2f}%")
        else:
            st.error("No faces detected in the video.")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

else:
    st.sidebar.write("Upload a video file to get started.")
