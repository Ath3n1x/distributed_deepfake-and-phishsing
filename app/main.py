import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
from scipy import signal
import librosa
import torch.nn.functional as F
import requests
import json
import os
import sys
import base64
import io
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.gradcam import GradCAM
from models.cnn_model import create_model
from utils.hash_match import match_hash
from data.db_manager import DatabaseManager
import imagehash
import random
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit UI Setup
st.set_page_config(page_title="DeepTrace Pro", layout="wide")
st.title("🔍 DeepTrace Pro - Advanced Deepfake Detection")

# Server configuration
SERVER_URL = os.getenv('SERVER_URL', 'http://localhost:8000')
server_url = st.sidebar.text_input("Server URL", value=SERVER_URL, key="phash_server_url")

# Add continuous analysis mode
st.sidebar.title("Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["Single Frame", "Advanced Analysis", "Distributed Analysis"]
)

# Node configuration
if analysis_mode == "Distributed Analysis":
    st.sidebar.subheader("Node Configuration")
    node_id = st.sidebar.text_input("Node ID", value=f"node_{random.randint(1000, 9999)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model and Processor
model, processor = create_model(device=device)
model.eval()

# GradCAM setup
def find_last_conv_layer(model):
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv

target_layer = find_last_conv_layer(model)
gradcam = GradCAM(model, target_layer=target_layer)

# DB setup
db = DatabaseManager("deeptrace.db")

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def get_frame_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not accessible.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("Failed to read from webcam.")
        return None
    return frame

def send_frame_to_server(frame, node_id):
    """Send frame to distributed server for analysis"""
    try:
        # Convert frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Prepare files and data
        files = {
            'file': ('frame.jpg', frame_bytes, 'image/jpeg')
        }
        data = {
            'node_id': node_id
        }
        
        # Send request
        response = requests.post(
            f"{server_url}/analyze",
            files=files,
            data=data
        )
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        st.error(f"Error sending frame to server: {str(e)}")
        return None

def get_task_result(task_id):
    """Get the result of a distributed analysis task"""
    try:
        response = requests.get(f"{server_url}/task/{task_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting task result: {str(e)}")
        return None

def infer_and_display(frame, use_distributed=False, node_id=None):
    if use_distributed:
        # Send frame to server
        result = send_frame_to_server(frame, node_id)
        if result and 'task_id' in result:
            st.info(f"Task queued: {result['task_id']}")
            
            # Poll for results
            with st.spinner("Waiting for analysis results..."):
                while True:
                    task_result = get_task_result(result['task_id'])
                    if task_result and 'status' not in task_result:
                        break
                    time.sleep(0.5)
            
            if task_result:
                st.subheader("Distributed Analysis Result")
                st.write(f"Node ID: {task_result.get('node_id', 'unknown')}")
                st.write(f"Confidence: {task_result.get('confidence', 0.0):.2f}")
                st.write(f"Is Fake: {task_result.get('is_fake', False)}")
                # Display original frame if available
                if 'frame_data' in task_result:
                    try:
                        frame_bytes = base64.b64decode(task_result['frame_data'])
                        frame_img = Image.open(io.BytesIO(frame_bytes))
                        st.image(frame_img, caption="Original Frame")
                    except Exception as e:
                        st.warning(f"Could not display original frame: {e}")
                # Display attention map if available and not None/empty
                attn_map = task_result.get('attention_map')
                if attn_map:
                    try:
                        attn_bytes = base64.b64decode(attn_map)
                        attn_img = Image.open(io.BytesIO(attn_bytes))
                        st.image(attn_img, caption="Attention Map")
                    except Exception as e:
                        st.warning(f"Could not display attention map: {e}")
                # Display hash metadata if available
                if 'hash_metadata' in task_result:
                    st.write("Hash Metadata:", task_result['hash_metadata'])
    else:
        # Local inference (existing code)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.max(probs, dim=1)[0].item()
            is_fake = pred > 0.5

        cam_output = gradcam.generate_cam(inputs['pixel_values'])[0]
        cam_output = cv2.resize(cam_output, (frame.shape[1], frame.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_output), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        hash_val = str(imagehash.phash(pil_img))
        matches = db.get_matches(hash_val)
        frame_id = db.add_frame(hash_val, pred, is_fake)

        st.subheader("Inference Result")
        st.image(overlay, caption=f"{'FAKE' if is_fake else 'REAL'} (Confidence: {pred:.2f})", channels="BGR")

        with st.expander("Hash Matches & Metadata"):
            st.write(f"pHash: `{hash_val}`")
            if matches:
                st.success(f"Found {len(matches)} similar frame(s) in DB:")
                for match in matches:
                    st.write(match)
            else:
                st.info("No previous matches found for this frame.")

def predict_frame(model, image, device):
    try:
        # Convert image to RGB if it's not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image using the Hugging Face processor
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.max(probs, dim=1)[0].item()
        
        return pred
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return 0.0

class ResearchMetrics:
    def __init__(self):
        self.metrics_history = []
        self.confidence_scores = []
        self.temporal_scores = []
        self.expression_scores = []
        self.attention_maps = []
        
    def calculate_metrics(self, frame, pred, attention_map=None):
        metrics = {
            'timestamp': datetime.now(),
            'deepfake_probability': pred,
            'temporal_consistency': 1.0,  # Placeholder
            'expression_consistency': 1.0,  # Placeholder
        }
        
        if attention_map is not None:
            metrics['attention_entropy'] = self.calculate_attention_entropy(attention_map)
            metrics['attention_focus'] = self.calculate_attention_focus(attention_map)
        
        self.metrics_history.append(metrics)
        return metrics
    
    def calculate_attention_entropy(self, attention_map):
        hist = cv2.calcHist([attention_map], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def calculate_attention_focus(self, attention_map):
        # Ensure attention_map is 2D
        if attention_map.ndim == 1:
            # Try to make it square if possible
            size = int(np.sqrt(attention_map.shape[0]))
            if size * size == attention_map.shape[0]:
                attention_map = attention_map.reshape((size, size))
            else:
                # Fallback: expand dims
                attention_map = np.expand_dims(attention_map, axis=0)
        elif attention_map.ndim > 2:
            attention_map = attention_map.squeeze()
            if attention_map.ndim > 2:
                attention_map = attention_map[0]
        center = np.array(attention_map.shape) / 2
        y, x = np.indices(attention_map.shape)
        dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        weighted_dist = attention_map * dist
        return 1.0 - (weighted_dist.sum() / (attention_map.sum() + 1e-10))
    
    def get_metrics_summary(self):
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        return {
            'mean_confidence': df['deepfake_probability'].mean(),
            'std_confidence': df['deepfake_probability'].std(),
            'mean_temporal': df['temporal_consistency'].mean(),
            'mean_expression': df['expression_consistency'].mean()
        }

class EnhancedVisualizer:
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_attention_heatmap(self, frame, attention_map):
        attention_map = cv2.resize(attention_map, (frame.shape[1], frame.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        return overlay
    
    def create_metrics_dashboard(self, metrics_history):
        df = pd.DataFrame(metrics_history)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['deepfake_probability'],
            name='Deepfake Probability',
            line=dict(color=self.colors[0])
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['temporal_consistency'],
            name='Temporal Consistency',
            line=dict(color=self.colors[1])
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['expression_consistency'],
            name='Expression Consistency',
            line=dict(color=self.colors[2])
        ))
        
        fig.update_layout(
            title='Multi-Modal Analysis Dashboard',
            xaxis_title='Time',
            yaxis_title='Score',
            hovermode='x unified'
        )
        
        return fig
    
    def create_attention_analysis(self, attention_maps):
        if not attention_maps:
            return None
        
        fig = go.Figure()
        
        for i, attention_map in enumerate(attention_maps[-5:]):
            hist = cv2.calcHist([attention_map], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            
            fig.add_trace(go.Scatter(
                y=hist.flatten(),
                name=f'Frame {i+1}',
                line=dict(color=self.colors[i % len(self.colors)])
            ))
        
        fig.update_layout(
            title='Attention Distribution Over Time',
            xaxis_title='Attention Intensity',
            yaxis_title='Frequency',
            showlegend=True
        )
        
        return fig

# Initialize components
research_metrics = ResearchMetrics()
visualizer = EnhancedVisualizer()

# Main analysis loop
if analysis_mode == "Distributed Analysis":
    st.subheader("Distributed Analysis Mode")
    if st.button("Start Distributed Analysis"):
        frame = get_frame_from_webcam()
        if frame is not None:
            infer_and_display(frame, use_distributed=True, node_id=node_id)
else:
    # Original analysis modes
    if analysis_mode == "Single Frame":
        st.subheader("Single Frame Analysis Mode")
        st.caption("Upload an image or capture from webcam for local analysis. pHash and match log will be shown.")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="main_upload")
        if st.button("Capture and Analyze"):
            frame = get_frame_from_webcam()
            if frame is not None:
                # --- Static Grad-CAM Visualizations ---
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                inputs = processor(images=pil_img, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    pred = torch.max(probs, dim=1)[0].item()
                    is_fake = pred > 0.5
                cam_output = gradcam.generate_cam(inputs['pixel_values'])[0]
                cam_output = cv2.resize(cam_output, (frame.shape[1], frame.shape[0]))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_output), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
                st.image(overlay, caption=f"Grad-CAM Overlay ({'FAKE' if is_fake else 'REAL'}, Confidence: {pred:.2f})", channels="BGR")
                st.image(cam_output, caption="Raw Grad-CAM Heatmap", channels="GRAY")
                st.bar_chart(cam_output.flatten())
                max_idx = np.unravel_index(np.argmax(cam_output), cam_output.shape)
                highlight_img = overlay.copy()
                cv2.circle(highlight_img, (max_idx[1], max_idx[0]), 10, (0,255,0), 2)
                st.image(highlight_img, caption="Max Impact Area Highlighted", channels="BGR")
                entropy = -np.sum(cam_output * np.log2(cam_output + 1e-10))
                focus = np.max(cam_output)
                mean = np.mean(cam_output)
                std = np.std(cam_output)
                st.write({"Attention Entropy": float(entropy), "Focus (max)": float(focus), "Mean": float(mean), "Std": float(std)})
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # --- Static Grad-CAM Visualizations for upload ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            inputs = processor(images=pil_img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred = torch.max(probs, dim=1)[0].item()
                is_fake = pred > 0.5
            cam_output = gradcam.generate_cam(inputs['pixel_values'])[0]
            cam_output = cv2.resize(cam_output, (frame.shape[1], frame.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_output), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
            st.image(overlay, caption=f"Grad-CAM Overlay ({'FAKE' if is_fake else 'REAL'}, Confidence: {pred:.2f})", channels="BGR")
            st.image(cam_output, caption="Raw Grad-CAM Heatmap", channels="GRAY")
            st.bar_chart(cam_output.flatten())
            max_idx = np.unravel_index(np.argmax(cam_output), cam_output.shape)
            highlight_img = overlay.copy()
            cv2.circle(highlight_img, (max_idx[1], max_idx[0]), 10, (0,255,0), 2)
            st.image(highlight_img, caption="Max Impact Area Highlighted", channels="BGR")
            entropy = -np.sum(cam_output * np.log2(cam_output + 1e-10))
            focus = np.max(cam_output)
            mean = np.mean(cam_output)
            std = np.std(cam_output)
            st.write({"Attention Entropy": float(entropy), "Focus (max)": float(focus), "Mean": float(mean), "Std": float(std)})
    elif analysis_mode == "Advanced Analysis":
        st.subheader("Advanced Analysis Mode")
        if 'running' not in st.session_state:
            st.session_state['running'] = False
        if 'adv_results' not in st.session_state:
            st.session_state['adv_results'] = []
        if not st.session_state['running']:
            if st.button("Start Advanced Analysis"):
                st.session_state['running'] = True
        else:
            if st.button("Stop Analysis"):
                st.session_state['running'] = False
        while st.session_state['running']:
            frame = get_frame_from_webcam()
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                inputs = processor(images=pil_img, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    pred = torch.max(probs, dim=1)[0].item()
                    is_fake = pred > 0.5
                hash_val = str(imagehash.phash(pil_img))
                # Display results
                st.image(frame, caption="Captured Frame", channels="BGR")
                st.write(f"pHash: `{hash_val}`")
                st.write({"Confidence": float(pred), "Is Fake": bool(is_fake)})
                st.write({"Softmax Probabilities": probs.cpu().numpy().tolist()})
                # Log results
                st.session_state['adv_results'].append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'pHash': hash_val,
                    'confidence': float(pred),
                    'is_fake': bool(is_fake)
                })
                # Show running table/log
                df = pd.DataFrame(st.session_state['adv_results'])
                st.dataframe(df.tail(10), use_container_width=True)
            time.sleep(2)  # Camera capture every 2 seconds

# --- Frame History and Upload ---

st.sidebar.title("Upload or View History")

# Upload image for single prediction
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="sidebar_upload")
if uploaded_file is not None:
    try:
        st.subheader("Uploaded Image Inference")
        image = Image.open(uploaded_file)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        st.image(image, caption="Uploaded Image")
        prediction = predict_frame(model, image, device)
        st.success(f"Prediction: {'Fake' if prediction > 0.5 else 'Real'} (Confidence: {prediction:.2f})")
    except Exception as e:
        st.error(f"Error processing uploaded image: {str(e)}")

# Show DB Stats
if st.sidebar.button("View Frame Log Summary"):
    stats = db.get_stats()
    st.subheader("📊 Frame Database Summary")
    st.write(stats)

# Show Recent Frames
if st.sidebar.button("Show Last 10 Frames"):
    import sqlite3
    import pandas as pd
    with sqlite3.connect("deeptrace.db") as conn:
        df = pd.read_sql_query("SELECT * FROM frames ORDER BY timestamp DESC LIMIT 10", conn)
        st.write(df)

# --- pHash Search UI ---
st.sidebar.subheader("pHash Search")
phash_query = st.sidebar.text_input("Enter pHash to search for matches")
phash_threshold = st.sidebar.number_input("Hamming Distance Threshold", min_value=0, max_value=64, value=5, step=1)
if st.sidebar.button("Search pHash") and phash_query:
    response = requests.post(f"{server_url}/match", json={"phash": phash_query, "threshold": phash_threshold})
    if response.ok:
        matches = response.json().get('matches', [])
        if matches:
            st.subheader("pHash Matches")
            df = pd.DataFrame(matches)
            st.dataframe(df[["hash", "confidence", "is_fake", "timestamp", "hamming_distance"]])
        else:
            st.warning("No matches found.")
    else:
        st.warning("Error in search.")

# --- Inference Result Visualization ---
def show_frame_result(result):
    st.subheader("Frame Analysis Result")
    st.write({k: v for k, v in result.items() if k not in ['attention_map', 'raw_attention_map', 'frame_data']})
    # Download all metadata as CSV
    if st.button("Download Metadata as CSV"):
        csv_str = "\n".join([f"{k},{v}" for k, v in result.items()])
        st.download_button("Download CSV", csv_str, file_name="frame_metadata.csv")
    # Show overlay and raw attention map
    if 'frame_data' in result:
        st.image(Image.open(io.BytesIO(base64.b64decode(result['frame_data']))), caption="Original Frame")
    if 'attention_map' in result:
        st.image(Image.open(io.BytesIO(base64.b64decode(result['attention_map']))), caption="Grad-CAM Overlay")
        st.download_button("Download Grad-CAM Overlay", base64.b64decode(result['attention_map']), file_name="gradcam_overlay.png")
    if 'raw_attention_map' in result:
        st.image(Image.open(io.BytesIO(base64.b64decode(result['raw_attention_map']))), caption="Raw Grad-CAM Map")
        st.download_button("Download Raw Grad-CAM Map", base64.b64decode(result['raw_attention_map']), file_name="raw_gradcam.png")
    if 'attention_stats' in result:
        st.write("Attention Map Stats:")
        st.json(result['attention_stats'])
    # Show softmax probabilities if available
    if 'probs' in result:
        st.write("Softmax Probabilities:")
        st.json(result['probs'])

# --- Frame Log Table and Export ---
if st.sidebar.button("Export Frame Log as CSV"):
    log = fetch_live_log(server_url)
    if log:
        keys = log[0].keys()
        with open("frame_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(log)
        st.success("Frame log exported as frame_log.csv")

# --- Advanced Research Metrics ---
# (Placeholder: add ROC, confusion matrix, etc. as needed)
# You can add a section to upload ground truth and compute ROC, etc.
