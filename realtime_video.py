import cv2
from ultralytics import YOLO
import numpy as np
import torch
from torch.nn.functional import interpolate

def start_realtime_feed():
    # Initialize YOLO model
    try:
        model = YOLO("yolov8n.pt", task='detect')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize MiDaS
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to('cuda' if torch.cuda.is_available() else 'cpu')
    midas.eval()
    
    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    # Initialize cameras (only need main camera now)
    main_cam = cv2.VideoCapture(0)
    
    # Set consistent resolution
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    main_cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    main_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Set up display windows
    cv2.namedWindow('Main Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)

    def process_frame(frame):
        # Resize frame to ensure consistent dimensions
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
        return edges, frame

    def compute_depth(frame):
        try:
            # Transform input for MiDaS
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(img).to('cuda' if torch.cuda.is_available() else 'cpu')

            # Compute depth
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(FRAME_HEIGHT, FRAME_WIDTH),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()
            
            # Normalize depth map
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Create color visualization
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
        # Convert edges to color (BGR) for overlay
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Create overlay
        overlay = cv2.addWeighted(depth_map, alpha, edges_color, 1-alpha, 0)
        return overlay

    while True:
        # Capture frame
        ret_main, main_frame = main_cam.read()
        
        if not ret_main:
            print("Error: Could not read from camera")
            break
            
        # Process frame
        main_edges, main_frame_resized = process_frame(main_frame)
        
        # Compute depth using MiDaS
        depth_map, depth_visualization = compute_depth(main_frame_resized)
        
        # Create overlay of edges and depth
        combined_overlay = create_overlay(main_edges, depth_visualization)
        
        # Run YOLO detection on main frame
        results = model(main_frame_resized, show=False)
        
        # Draw detection boxes
        for result in results:
            annotated_frame = result.plot()
            
            # Display frames
            cv2.imshow('Main Camera', annotated_frame)
            cv2.imshow('Edge Detection', main_edges)
            cv2.imshow('Depth Map', depth_visualization)
            cv2.imshow('Overlay', combined_overlay)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    main_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_realtime_feed()