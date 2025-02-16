import cv2
from ultralytics import YOLO
import numpy as np
import torch
from camera_thread import ThreadedCamera
from discretize import Discretizer
import time
from point_cloud import CloudRenderer
import requests
import json

def start_realtime_feed():
    # Initialize YOLO model with better performance settings
    try:
        model = YOLO("yolov8n.pt", task='detect')
        model.fuse()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize MiDaS with smaller model
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        midas.to(device)
        midas.eval()
        
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True
    except Exception as e:
        print(f"Error loading MiDaS: {e}")
        return
    
    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    # Initialize threaded camera
    FRAME_WIDTH = 480
    FRAME_HEIGHT = 360
    main_cam = ThreadedCamera(0, FRAME_WIDTH, FRAME_HEIGHT)

    # Set up display windows
    cv2.namedWindow('Main Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)

    def process_frame(frame):
        # Optimize edge detection parameters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        return edges, frame

    def compute_depth(frame):
        try:
            # Transform input for MiDaS
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(img).to(device)

            # Compute depth with optimized settings
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(FRAME_HEIGHT, FRAME_WIDTH),
                    mode="bilinear",  # Changed from bicubic to bilinear for speed
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
            
            return depth_map, depth_color
            
        except Exception as e:
            print(f"Error in depth computation: {e}")
            blank = np.zeros_like(frame)
            return blank, blank

    def create_overlay(edges, depth_map, alpha=0.7):
        """
        Create an overlay of edge detection and depth map
        Args:
            edges: Edge detection image (grayscale)
            depth_map: Depth visualization (color)
            alpha: Transparency factor (0.0 to 1.0)
        """
        # Ensure both images have the same dimensions
        edges = cv2.resize(edges, (depth_map.shape[1], depth_map.shape[0]))
        
        # Convert edges to color (BGR) for overlay
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Ensure both images have the same data type
        edges_color = edges_color.astype(np.uint8)
        depth_map = depth_map.astype(np.uint8)
        
        # Create overlay
        overlay = cv2.addWeighted(depth_map, alpha, edges_color, 1-alpha, 0)
        return overlay

    # Initialize discretizer
    discretizer = Discretizer(rows=20, cols=40)
    last_discretize_time = time.time()
    DISCRETIZE_INTERVAL = 2.0  # Seconds between discretization

    # Initialize CloudRenderer
    cloud_renderer = CloudRenderer(FRAME_WIDTH, FRAME_HEIGHT)
    last_cloud_update = time.time()
    CLOUD_UPDATE_INTERVAL = 2.0  # Update point cloud every X seconds

    def send_point_cloud(points):
        """Send point cloud data to API with detailed logging"""
        try:
            # Log the data being sent
            print("\nSending Point Cloud Data:")
            print(f"Shape of point cloud: {points.shape}")
            print(f"Sample of points being sent:")
            print(points[:5] if len(points) > 5 else points)
            
            response = requests.post(
                'http://localhost:5000/point_cloud',
                json={'point_cloud': points.tolist()},
                headers={'Content-Type': 'application/json'}
            )
            
            # Save to JSON file
            try:
                json_data = {
                    'point_cloud': points.tolist()
                }
                with open('test_point_cloud.json', 'w') as f:
                    json.dump(json_data, f, indent=4)
                print("\nSuccessfully saved point cloud to test_point_cloud.json")
                print(f"Saved {len(points)} points")
            except Exception as e:
                print(f"\nError saving to JSON file: {e}")
            
            # Log the API response
            if response.status_code == 200:
                response_data = response.json()
                print("\nAPI Response:")
                print(f"Status: SUCCESS")
                print(f"Points successfully sent: {len(points)}")
                print(f"Server received points: {response_data.get('points_received')}")
            else:
                print("\nAPI Response:")
                print(f"Status: FAILED")
                print(f"Status code: {response.status_code}")
                print(f"Error message: {response.text}")
            
        except requests.exceptions.ConnectionError:
            print("\nERROR: Could not connect to API server")
            print("Make sure the Flask server (app.py) is running on http://localhost:5000")
        except Exception as e:
            print(f"\nERROR: Unexpected error while sending point cloud data:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")

    # Process frames in main loop
    frame_count = 0
    while True:
        # Skip frames for performance (process every 2nd frame)
        ret_main, main_frame = main_cam.read()
        frame_count += 1
        
        if not ret_main:
            print("Error: Could not read from camera")
            break
            
        if frame_count % 2 != 0:  # Process every other frame
            continue
            
        # Process frame
        main_edges, main_frame_resized = process_frame(main_frame)
        
        # Compute depth using MiDaS
        depth_map, depth_visualization = compute_depth(main_frame_resized)
        
        # Create overlay of edges and depth
        combined_overlay = create_overlay(main_edges, depth_visualization)
        
        # Discretize at regular intervals
        current_time = time.time()
        if current_time - last_discretize_time >= DISCRETIZE_INTERVAL:
            discretizer.add_frame(main_edges, depth_map)
            discretizer.print_latest()
            last_discretize_time = current_time
        
        # Update point cloud at regular intervals
        if current_time - last_cloud_update >= CLOUD_UPDATE_INTERVAL:
            print("\n" + "="*50)
            print("Point Cloud Update at time:", time.strftime("%H:%M:%S"))
            print("="*50)
            
            # Update depth point cloud and send to API
            depth_points = cloud_renderer.update_depth_cloud(depth_map)
            if depth_points is not None and len(depth_points) > 0:
                print("\nAttempting to send depth point cloud to API...")
                send_point_cloud(depth_points)
            else:
                print("\nNo valid depth points generated to send")

            
            # Update edge point cloud (if needed)
            cloud_renderer.update_depth_cloud(depth_map)
            cloud_renderer.update_edge_cloud(main_edges)
            
            last_cloud_update = current_time
        
        # Update point cloud visualization
        if cloud_renderer.is_running:
            if not cloud_renderer.update_visualization():
                print("Failed to update visualization")
                break
        
        # Run YOLO detection with optimized settings
        results = model(main_frame_resized, conf=0.5, iou=0.45)  # Adjusted confidence thresholds
        
        # Draw detection boxes
        for result in results:
            annotated_frame = result.plot()
            
            # Display frames
            # cv2.imshow('Main Camera', annotated_frame)
            # cv2.imshow('Edge Detection', main_edges)
            # cv2.imshow('Depth Map', depth_visualization)
            cv2.imshow('Overlay', combined_overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save or process final data if needed
    final_data = discretizer.get_history()
    # You could save final_data here if needed
    
    # Cleanup
    cloud_renderer.close()
    main_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_realtime_feed()