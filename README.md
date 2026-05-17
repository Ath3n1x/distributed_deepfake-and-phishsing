# DeepTrace Pro - Distributed Deepfake Detection

## Overview

**DeepTrace Pro** is a distributed, real-time deepfake detection system with a modern Streamlit frontend and a FastAPI backend. It supports both single-node and multi-node (distributed) operation, providing advanced features for deepfake analysis, Grad-CAM visualization, and frame history management.

## Features

- **Distributed Analysis:**
  - Multiple nodes (clients) can send frames to a central FastAPI backend for deepfake analysis.
  - Each node can be uniquely identified via a Node ID.
  - Results include prediction, confidence, attention (Grad-CAM) map, and node metadata.

- **Single Frame & Advanced Analysis:**
  - Analyze a single frame from webcam or upload.
  - Advanced/continuous analysis mode for repeated or research-oriented inference.

- **Grad-CAM Visualization:**
  - Attention maps (Grad-CAM overlays) are generated and displayed for each analyzed frame.

- **Frame Upload & Webcam:**
  - Upload images (JPG, JPEG, PNG) for analysis.
  - Capture frames directly from webcam.

- **Frame Log & History:**
  - View last 10 analyzed frames and summary statistics from the local database.
  - Distributed log shows results from all nodes.

- **Robust Error Handling:**
  - Handles missing images, backend errors, and import issues gracefully.

## New Features for Research & Demo

- **/match Endpoint:**
  - POST endpoint to search for frames by pHash, returns all matches from the database.
- **pHash Search UI:**
  - Sidebar input for pHash, displays all matching frames and metadata.
- **Advanced Grad-CAM Visualization:**
  - Shows both overlay and raw Grad-CAM map for each frame.
  - Displays attention map statistics (entropy, focus, mean, std).
  - Download buttons for overlay and raw map images.
- **Frame Metadata Table & Export:**
  - All frame metadata shown in a table, with CSV export option.
- **Advanced Research Metrics:**
  - Softmax probabilities, ROC curve, and log export for research paper results.

## Quick Start

### Windows (Recommended)
1. **Setup (one-time):**
   ```cmd
   setup.bat
   ```

2. **Run the application:**
   ```cmd
   run.bat
   ```

### Manual Setup

1. **Create and activate virtual environment:**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate.bat  # Windows
   source .venv/bin/activate   # macOS/Linux
   ```

2. **Install minimal dependencies:**
   ```sh
   pip install -r requirements-minimal.txt
   ```

3. **Start the backend:**
   ```sh
   uvicorn app.api.server:app --host 0.0.0.0 --port 8000
   ```

4. **Start the frontend:**
   ```sh
   streamlit run app/main.py
   ```

## Project Structure

```
DeepTrace_Streamlit_Fmaybe/
├── app/                    # Main application
│   ├── api/               # FastAPI backend
│   ├── client/            # Node client
│   ├── worker/            # Celery worker
│   └── main.py            # Streamlit frontend
├── models/                # CNN models
├── utils/                 # Utilities (GradCAM, hash matching)
├── data/                  # Database management
├── train/                 # Training scripts
├── .venv/                 # Virtual environment (created by setup)
├── requirements-minimal.txt # Minimal dependencies (~416 MB total)
├── requirements.txt       # Full dependencies (if needed)
├── setup.bat             # Windows setup script
├── run.bat               # Windows run script
└── README.md             # This file
```

## Requirements
- Python 3.8+
- See `requirements-minimal.txt` for essential dependencies

## File Size Optimization

The project has been heavily optimized to reduce file size.

## Notes
- Always run both backend and frontend from the project root directory.
- For distributed mode, multiple machines can connect to the same backend by specifying its IP and port.
- The virtual environment (`.venv/`) is excluded from version control to keep repository size small.
- If you need additional packages, use `requirements.txt` instead of `requirements-minimal.txt`.

## License
MIT
