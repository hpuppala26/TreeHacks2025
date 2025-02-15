import cv2
from ultralytics import YOLO
import numpy as np

def start_realtime_feed():
    # Initialize YOLO model
    try:
        model = YOLO("yolov8n.pt", task='detect')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

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
    
    # Modify stereo matcher parameters for better depth perception
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=256,  # Increased from 128 for wider depth range
        blockSize=5,         # Reduced for better detail
        P1=8 * 3 * 5**2,    # Adjusted for new blockSize
        P2=32 * 3 * 5**2,   # Adjusted for new blockSize
        disp12MaxDiff=1,
        uniquenessRatio=10,  # Reduced to allow more matches
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Modify WLS filter parameters
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(8000)    # Controls smoothness
    wls_filter.setSigmaColor(1.2)  # Controls color-dependent filtering
    
    # Additional WLS parameters
    wls_filter.setLRCthresh(24)    # Left-right consistency check threshold
    wls_filter.setDepthDiscontinuityRadius(7)  # Radius for depth discontinuity detection

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
        
        # Apply bilateral filter with adjusted parameters
        left_filtered = cv2.bilateralFilter(left_gray, d=5, sigmaColor=50, sigmaSpace=50)
        right_filtered = cv2.bilateralFilter(right_gray, d=5, sigmaColor=50, sigmaSpace=50)
        
        # Compute disparities
        left_disp = stereo.compute(left_filtered, right_filtered)
        right_disp = right_matcher.compute(right_filtered, left_filtered)
        
        # Convert to correct format
        left_disp = left_disp.astype(np.float32) / 16.0
        right_disp = right_disp.astype(np.float32) / 16.0
        
        # Apply WLS filtering
        filtered_disp = wls_filter.filter(left_disp, left_gray, disparity_map_right=right_disp)
        
        # Apply temporal smoothing with adjusted weights
        if not hasattr(compute_depth, 'previous_disp'):
            compute_depth.previous_disp = filtered_disp.copy()
        else:
            if compute_depth.previous_disp is not None:
                filtered_disp = (filtered_disp * 0.8 + compute_depth.previous_disp * 0.2)  # More weight to current frame
        
        compute_depth.previous_disp = filtered_disp.copy()
        
        # Normalize disparity with adjusted range
        disparity_normalized = cv2.normalize(filtered_disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_normalized = disparity_normalized.astype(np.uint8)
        
        # Optional: Invert the colormap if needed
        # disparity_normalized = 255 - disparity_normalized
        
        # Apply final smoothing
        disparity_normalized = cv2.medianBlur(disparity_normalized, 5)
        
        # Create color visualization
        disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
        
        return filtered_disp, disparity_color

    # Initialize the previous_disp attribute
    compute_depth.previous_disp = None

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