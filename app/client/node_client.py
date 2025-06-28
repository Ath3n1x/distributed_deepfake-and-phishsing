import cv2
import requests
import base64
import json
import time
import uuid
from datetime import datetime
import os
from typing import Optional, Dict, Any

class DeepTraceNode:
    def __init__(self, server_url: str, node_id: Optional[str] = None):
        self.server_url = server_url
        self.node_id = node_id or str(uuid.uuid4())
        self.session = requests.Session()
        
    def capture_frame(self, source: int = 0) -> Optional[bytes]:
        """Capture a frame from webcam or video source"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source {source}")
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
            
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    
    def analyze_frame(self, frame_data: bytes) -> Dict[str, Any]:
        """Send frame to server for analysis"""
        try:
            # Prepare multipart form data
            files = {
                'file': ('frame.jpg', frame_data, 'image/jpeg')
            }
            data = {
                'node_id': self.node_id
            }
            
            # Send request
            response = self.session.post(
                f"{self.server_url}/analyze",
                files=files,
                data=data
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """Get the result of an analysis task"""
        try:
            response = self.session.get(f"{self.server_url}/task/{task_id}")
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def continuous_analysis(self, source: int = 0, interval: float = 1.0):
        """Continuously capture and analyze frames"""
        while True:
            try:
                # Capture frame
                frame_data = self.capture_frame(source)
                if frame_data is None:
                    print("Failed to capture frame")
                    continue
                
                # Send for analysis
                result = self.analyze_frame(frame_data)
                if 'task_id' in result:
                    print(f"Task queued: {result['task_id']}")
                    
                    # Poll for results
                    while True:
                        task_result = self.get_task_result(result['task_id'])
                        if 'status' not in task_result or task_result['status'] != 'processing':
                            print(f"Analysis result: {task_result}")
                            break
                        time.sleep(0.5)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("Stopping continuous analysis")
                break
            except Exception as e:
                print(f"Error during analysis: {e}")
                time.sleep(interval)

if __name__ == "__main__":
    # Example usage
    node = DeepTraceNode("http://localhost:8000")
    node.continuous_analysis()