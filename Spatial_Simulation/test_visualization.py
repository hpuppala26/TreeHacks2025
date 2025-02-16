from state import simState
import numpy as np
import os
import json

def main():
    # Create state and set initial conditions
    state = simState()
    
    # Set initial linear conditions
    state.velocity = np.array([0.0, 0.0, 0.0])  # start from rest
    state.acceleration = np.array([0.1, 0.05, 0.0])  # constant acceleration
    
    # Set initial angular conditions
    state.angular_velocity = np.array([0.5, 0.3, 0.1])  # initial rotation rates (rad/s)
    # state.angular_acceleration = np.array([0.02, 0.01, 0.0])  # constant angular acceleration
    state.angular_acceleration = np.zeros(3)  # Start with no rotation acceleration

    # Load point cloud data if available
    try:
        # Get the directory of the current script and go up one level
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # Go up one directory
        point_cloud_path = os.path.join(parent_dir, 'test_point_cloud.json')
        
        print(f"Looking for point cloud data at: {point_cloud_path}")
        
        with open(point_cloud_path, 'r') as f:
            point_cloud_data = json.load(f)
            print("Successfully loaded point cloud data")
            # You can now use point_cloud_data in your simulation
            
            # Add the point cloud data to the state
            if 'point_cloud' in point_cloud_data:
                points = np.array(point_cloud_data['point_cloud'])
                state.set_point_cloud(points)  # Assuming there's a method to set point cloud data
                print(f"Loaded {len(points)} points into simulation")
    except Exception as e:
        print(f"Error loading point cloud data: {e}")
    
    # Run animation
    state.animate_scene(num_frames=200)

if __name__ == "__main__":
    main() 