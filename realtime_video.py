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
    
    def process_frame(frame):
        # Resize frame to ensure consistent dimensions
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
        return edges, frame
    
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
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    main_cam.release()
    spatial_cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_realtime_feed()