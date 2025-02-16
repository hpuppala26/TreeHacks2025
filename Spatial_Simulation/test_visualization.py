from state import simState
import numpy as np

def main():
    # Create state and set initial conditions
    state = simState()
    
    # Set initial linear conditions
    state.velocity = np.array([0.0, 0.0, 0.0])  # start from rest
    state.acceleration = np.array([0.1, 0.05, 0.0])  # constant acceleration
    
    # Set initial angular conditions
    state.angular_velocity = np.array([0.5, 0.3, 0.1])  # initial rotation rates (rad/s)
    state.angular_acceleration = np.array([0.02, 0.01, 0.0])  # constant angular acceleration
    
    # Run animation
    state.animate_scene(num_frames=200)

if __name__ == "__main__":
    main() 