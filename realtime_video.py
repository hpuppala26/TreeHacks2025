import cv2
from ultralytics import YOLO
import numpy as np

def start_realtime_feed():
    # Initialize YOLO model
    model = YOLO("yolov8n.pt")
    
    # Initialize cameras
    main_cam = cv2.VideoCapture(0)
    spatial_cam = cv2.VideoCapture(1)
    
    # Set consistent resolution for both cameras
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # Set resolution for both cameras
    main_cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    main_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    spatial_cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    spatial_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Set up display windows
    cv2.namedWindow('Main Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Spatial Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
    
    # Initialize stereo matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # must be divisible by 16
        blockSize=9,
        P1=8 * 3 * 9 ** 2,  # First parameter controlling disparity smoothness
        P2=32 * 3 * 9 ** 2,  # Second parameter controlling disparity smoothness
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    def process_frame(frame):
        # Resize frame to ensure consistent dimensions
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
        return edges, frame
    
    def compute_depth(left_frame, right_frame):
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Normalize disparity for visualization
        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap for better visualization
        disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
        
        return disparity, disparity_color

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

    # Create new window for depth map
    cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)

    while True:
        # Capture frames
        ret_main, main_frame = main_cam.read()
        ret_spatial, spatial_frame = spatial_cam.read()
        
        if not (ret_main and ret_spatial):
            print("Error: Could not read from one or both cameras")
            break
            
        # Process frames
        main_edges, main_frame_resized = process_frame(main_frame)
        spatial_edges, spatial_frame_resized = process_frame(spatial_frame)
        
        # Compute depth
        disparity, depth_visualization = compute_depth(main_frame_resized, spatial_frame_resized)
        
        # Create overlay of edges and depth
        combined_overlay = create_overlay(main_edges, depth_visualization)
        
        # Run YOLO detection on main frame
        results = model(main_frame_resized, show=False)
        
        # Draw detection boxes
        for result in results:
            annotated_frame = result.plot()
            
            # Display frames
            cv2.imshow('Main Camera', annotated_frame)
            cv2.imshow('Spatial Camera', spatial_frame_resized)
            
            # Display edge detection
            combined_edges = np.hstack((main_edges, spatial_edges))
            cv2.imshow('Edge Detection', combined_edges)
            cv2.imshow('Depth Map', depth_visualization)
            cv2.imshow('Overlay', combined_overlay)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    main_cam.release()
    spatial_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_realtime_feed()