import numpy as np


# Define State Classs
class simState:
    
    # Define Time Variables:
    start_time: float
    time_step: float
    current_time: float
    end_time: float
    
    # Define State Variables:
    position: np.ndarray
    velocity: np.ndarray
    
    # Define an array that defines the point cloud of the primary object:
    primary_object_point_cloud: np.ndarray
    
    # Define an array that defines the point cloud of all surrounding objects:
    surrounding_objects_point_cloud: np.ndarray
    
    
    
    def propagate_dynamics_primary_object(self):
        '''
        Propagate the dynamics of the simulation
        Propagate the position and velocity of the primary object
        '''
        while self.current_time < self.end_time:
            self.current_time += self.time_step
            
            # Propagate the position of the primary object:
            self.position += self.velocity * self.time_step
            
            # Propagate the velocity of the primary object:
            self.velocity += self.acceleration * self.time_step
        
    
    