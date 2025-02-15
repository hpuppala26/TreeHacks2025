from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
import time
from ultralytics import YOLO
import random

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLO model
model = YOLO("yolov8n.pt")  # Small YOLOv8 model

@app.route("/")
def home():
    return "Airplane Obstacle Avoidance System Running!"

@app.route("/detect", methods=["POST"])
def detect_obstacles():
    try:
        file = request.files["image"]
        image = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Run YOLO detection
        results = model(img)
        detected_objects = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = result.names[int(box.cls[0])]

                detected_objects.append({"object": label, "confidence": confidence, "bbox": [x1, y1, x2, y2]})

        # Send obstacle data to dashboard
        socketio.emit("obstacle_data", {"objects": detected_objects})

        return jsonify({"status": "success", "objects": detected_objects})

    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Auto-send obstacle data every 5 seconds
def auto_send_fake_obstacles():
    while True:
        fake_data = {
            "objects": [
                {"object": "helicopter", "bbox": [random.randint(50, 750), random.randint(50, 400), random.randint(50, 750) + 50, random.randint(50, 400) + 50]},
                {"object": "fleet_of_birds", "bbox": [random.randint(50, 750), random.randint(50, 400), random.randint(50, 750) + 50, random.randint(50, 400) + 50]},
                {"object": "aeroplane", "bbox": [random.randint(50, 750), random.randint(50, 400), random.randint(50, 750) + 50, random.randint(50, 400) + 50]},
            ]
        }


        print("Sending fake obstacle data:", fake_data)
        socketio.emit("obstacle_data", fake_data)
        time.sleep(5)  # Send data every 5 seconds

# Run the auto data sender in the background
import threading
threading.Thread(target=auto_send_fake_obstacles, daemon=True).start()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
