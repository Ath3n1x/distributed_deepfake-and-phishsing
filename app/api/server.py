from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import io
from PIL import Image
import numpy as np
import torch
import json
from datetime import datetime
import os
import threading
import queue
import time
import uuid
import sys
import cv2
from utils.gradcam import GradCAM
from models.cnn_model import create_model
import torch.nn as nn
import sqlite3
import imagehash

# Initialize FastAPI app
app = FastAPI(title="DeepTrace Distributed API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task queue and results storage
task_queue = queue.Queue()
task_results = {}
task_lock = threading.Lock()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, processor = create_model(device=device)
model.eval()

def find_last_conv_layer(model):
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv

target_layer = find_last_conv_layer(model)
gradcam = GradCAM(model, target_layer=target_layer)

def compute_attention_stats(attention_map):
    # attention_map is a numpy array (float, 0-1)
    hist = np.histogram(attention_map, bins=256, range=(0,1), density=True)[0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    focus = np.max(attention_map)
    mean = np.mean(attention_map)
    std = np.std(attention_map)
    return {'entropy': float(entropy), 'focus': float(focus), 'mean': float(mean), 'std': float(std)}

def process_frame(frame_data, metadata):
    try:
        node_id = metadata.get('node_id') or 'unnamed_node'
        # Convert base64 to image
        image_data = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        # Preprocess
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.max(probs, dim=1)[0].item()
            is_fake = pred > 0.5
        # GradCAM
        cam_output = gradcam.generate_cam(inputs['pixel_values'])[0]
        cam_output = cv2.resize(cam_output, (image.width, image.height))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_output), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)
        # Encode overlay as base64 PNG
        overlay_img = Image.fromarray(overlay.astype('uint8'))
        buf = io.BytesIO()
        overlay_img.save(buf, format='PNG')
        attention_map_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        # Encode raw attention map as base64 PNG
        raw_map_img = Image.fromarray(np.uint8(255 * cam_output))
        raw_buf = io.BytesIO()
        raw_map_img.save(raw_buf, format='PNG')
        raw_attention_map_b64 = base64.b64encode(raw_buf.getvalue()).decode('utf-8')
        # Compute stats
        attn_stats = compute_attention_stats(cam_output)
        # Encode original frame as base64
        orig_buf = io.BytesIO()
        image.save(orig_buf, format='PNG')
        frame_data_b64 = base64.b64encode(orig_buf.getvalue()).decode('utf-8')
        result = {
            'is_fake': bool(is_fake),
            'confidence': float(pred),
            'attention_map': attention_map_b64,
            'raw_attention_map': raw_attention_map_b64,
            'attention_stats': attn_stats,
            'frame_data': frame_data_b64,
            'hash_metadata': None,  # Add hash logic if needed
            'timestamp': datetime.now().isoformat(),
            'node_id': node_id,
            'filename': metadata.get('filename', 'unknown'),
            'phash': str(imagehash.phash(image))
        }
        return result
    except Exception as e:
        return {'error': str(e)}

def worker():
    """Background worker to process tasks"""
    while True:
        try:
            task_id, frame_data, metadata = task_queue.get()
            result = process_frame(frame_data, metadata)
            with task_lock:
                task_results[task_id] = result
            task_queue.task_done()
        except Exception as e:
            print(f"Worker error: {e}")
        time.sleep(0.1)

# Start worker thread
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

@app.post("/analyze")
async def analyze_frame(
    file: UploadFile = File(...),
    node_id: str = Form("unknown")
):
    try:
        # Read image file
        contents = await file.read()
        base64_data = base64.b64encode(contents).decode()
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Prepare metadata
        metadata = {
            'node_id': node_id,
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename
        }
        
        # Add task to queue
        task_queue.put((task_id, base64_data, metadata))
        
        return JSONResponse({
            'task_id': task_id,
            'status': 'processing',
            'message': 'Frame queued for analysis'
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    with task_lock:
        if task_id in task_results:
            result = task_results[task_id]
            # Clean up old results
            if len(task_results) > 1000:  # Keep only last 1000 results
                old_tasks = list(task_results.keys())[:-1000]
                for old_task in old_tasks:
                    del task_results[old_task]
            return JSONResponse(result)
    
    return JSONResponse({
        'status': 'processing',
        'task_id': task_id
    })

@app.post("/match")
async def match_phash(phash: str = Body(...)):
    conn = sqlite3.connect("deeptrace.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, hash, confidence, is_fake, timestamp FROM frames WHERE hash = ?", (phash,))
    matches = cursor.fetchall()
    conn.close()
    # Return as list of dicts for frontend compatibility
    return {'matches': [
        {'id': row[0], 'hash': row[1], 'confidence': row[2], 'is_fake': row[3], 'timestamp': row[4]} for row in matches
    ]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 