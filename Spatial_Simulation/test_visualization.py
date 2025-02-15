from state import simState
import numpy as np

def main():
    # Create state and set initial conditions
    state = simState()
    
    # Set initial velocity and time parameters
    state.velocity = np.array([1.0, 0.5, 0.0])  # initial velocity
    state.dt = 0.1  # time step
    
    # Run animation
    state.animate_scene(num_frames=200)

if __name__ == "__main__":
    main() 