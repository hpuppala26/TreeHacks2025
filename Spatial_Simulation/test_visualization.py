from state import simState
import numpy as np
import time
import requests
import threading

def fetch_point_cloud(state):
    """
    Continuously fetch point cloud data from API
    """
    while True:
        try:
            response = requests.get('http://localhost:5000/point_cloud')
            if response.status_code == 200:
                data = response.json()
                if 'point_cloud' in data:
                    points = np.array(data['point_cloud'])
                    state.update_surrounding_point_cloud(points)
            time.sleep(0.1)  # Poll every 100ms
        except Exception as e:
            print(f"Error fetching point cloud: {e}")
            time.sleep(1)  # Wait longer on error

def main():
    # Create state and set initial conditions
    state = simState()
    
    # Set initial linear conditions
    state.velocity = np.array([0.0, 0.0, 0.0])  # start from rest
    state.acceleration = np.array([0, 0.05, 0.0])  # constant acceleration
    
    # Set initial angular conditions
    state.angular_velocity = np.array([0, 0, 0])  # initial rotation rates (rad/s)
    # state.angular_acceleration = np.array([0.02, 0.01, 0.0])  # constant angular acceleration
    state.angular_acceleration = np.zeros(3)  # Start with no rotation acceleration
    
     # Initialize world points
    n_world_points = 100
    self.world_points = np.random.uniform(-10, 10, (3, n_world_points))
    
    

    # Start point cloud update thread
    update_thread = threading.Thread(target=fetch_point_cloud, args=(state,), daemon=True)
    update_thread.start()
    
    # Run animation
    state.animate_scene(num_frames=200)

if __name__ == "__main__":
    main() 