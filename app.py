from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
import time
from ultralytics import YOLO
import random
import base64
import atexit
import json
from Spatial_Simulation.state import simState  # Add this import

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
current_point_cloud = None
sim_state = simState()  # Create instance of simState

# Load YOLO model (this can be changed to a larger object detection dataset from YOLO later on)
model = YOLO("yolov8n.pt")

# Initialize camera captures
main_cam = cv2.VideoCapture(0)  # Main camera
spatial_cam = cv2.VideoCapture(1)  # Spatial awareness camera

def process_frame(frame):
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
    
    return edges

@app.route("/")
def home():
    return "Airplane Obstacle Avoidance System Running!"

@app.route("/detect", methods=["POST"])
def detect_obstacles():
    try:
        # Capture frames from both cameras
        ret_main, main_frame = main_cam.read()
        ret_spatial, spatial_frame = spatial_cam.read()
        
        if not (ret_main and ret_spatial):
            raise Exception("Failed to capture from one or both cameras")

        # Process main camera frame
        main_edges = process_frame(main_frame)
        
        # Process spatial awareness camera frame
        spatial_edges = process_frame(spatial_frame)

        # Run YOLO detection on main frame
        results = model(main_frame)
        detected_objects = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = result.names[int(box.cls[0])]
                
                detected_objects.append({
                    "object": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

        # Encode processed frames to base64 for sending
        _, main_buffer = cv2.imencode('.jpg', main_edges)
        _, spatial_buffer = cv2.imencode('.jpg', spatial_edges)
        
        main_edges_b64 = base64.b64encode(main_buffer).decode('utf-8')
        spatial_edges_b64 = base64.b64encode(spatial_buffer).decode('utf-8')

        # Send processed data to dashboard
        socketio.emit("processed_data", {
            "objects": detected_objects,
            "main_edges": main_edges_b64,
            "spatial_edges": spatial_edges_b64
        })

        return jsonify({
            "status": "success",
            "objects": detected_objects
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/point_cloud", methods=["POST"])
def update_point_cloud():
    """Endpoint to receive and store point cloud data"""
    try:
        data = request.json
        point_cloud = np.array(data['point_cloud'])
        
        # Update both global and simulation state
        global current_point_cloud
        current_point_cloud = point_cloud
        
        # Update simulation state directly
        sim_state.update_surrounding_point_cloud(point_cloud)
        
        return jsonify({
            "status": "success", 
            "points_received": len(point_cloud),
            "last_five_points": point_cloud[-5:].tolist() if len(point_cloud) >= 5 else point_cloud.tolist()
        })
    except Exception as e:
        print(f"Error in update_point_cloud: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/point_cloud", methods=["GET"])
def get_point_cloud():
    """Endpoint to retrieve the latest point cloud data"""
    global current_point_cloud
    if current_point_cloud is not None:
        return jsonify({
            "status": "success",
            "point_cloud": current_point_cloud.tolist(),
            "shape": current_point_cloud.shape
        })
    return jsonify({"status": "no_data"}), 404

# Cleanup function
def cleanup():
    main_cam.release()
    spatial_cam.release()

# Register cleanup function
atexit.register(cleanup)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
