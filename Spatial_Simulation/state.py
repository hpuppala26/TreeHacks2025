import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import json
import time
import threading  

# Define State Class
class simState:
    
    def __init__(self):
        # Create a sphere for the primary object
        radius = 1.0  
        n_points = 1000  

        # Generate sphere coordinates
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        theta = np.arccos(np.random.uniform(-1, 1, n_points))

        # Convert to Cartesian coordinates (3xN array)
        self.primary_object_point_cloud = np.array([
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta)
        ])

        # Initialize physics parameters
        self.current_time = 0.0
        self.dt = 0.1
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.angular_velocity = np.array([0.0, 0.0, 0.1])
        self.angular_acceleration = np.zeros(3)

        # Fetch real-time sensor values at startup
        #self.acceleration, self.orientation = self.read_sensor_data()
        self.acceleration = np.array([0.0, 0.0, 0.0])   # DELETE THIS LATER
        self.orientation = np.array([0.0, 0.0, 0.0])   # DELETE THIS LATER
        print(f"🔄 INIT: Acceleration {self.acceleration}, Orientation {self.orientation}")

        # Initialize world points
        n_world_points = 100
        self.world_points = np.random.uniform(-10, 10, (3, n_world_points))
        
        # Initialize other attributes
        self.surrounding_objects_point_cloud = np.array([])
        self.primary_center = np.zeros(3)
        self.point_cloud = None

    def read_sensor_data(self):
        """
        Reads the latest sensor data from the JSON file in real-time.
        """
        while True:
            try:
                with open("/Users/sidharthanantha/Sidharth's Files/Stanford University/Hackathons/Treehacks 2025/TreeHacks2025/sensor_data.json", "r") as f:
                    sensor_data = json.load(f)

                # ✅ Debug info
                print(f"📡 LIVE SENSOR DATA -> {sensor_data}")

                acceleration = np.array([
                    sensor_data["acceleration"]["AccX"],
                    sensor_data["acceleration"]["AccY"],
                    sensor_data["acceleration"]["AccZ"]
                ])

                orientation = np.array([
                    sensor_data["orientation"]["Roll"],
                    sensor_data["orientation"]["Pitch"],
                    sensor_data["orientation"]["Yaw"]
                ])

                return acceleration, orientation

            except (FileNotFoundError, json.JSONDecodeError):
                print(f"⚠️ WARNING: Sensor file not found! Retrying...")
                time.sleep(0.1)

    def update_state(self):
        """
        Continuously updates the state with the latest sensor data.
        Runs in a background thread.
        """
        while True:
            #self.acceleration, self.orientation = self.read_sensor_data()

            # Apply dynamics update
            self.propagate_dynamics_primary_object()

            time.sleep(0.1)  # Control update rate (10Hz)

    def propagate_dynamics_primary_object(self):
        """
        Updates state variables using acceleration data in real-time.
        """
        self.current_time += self.dt

        # ✅ Apply acceleration updates
        self.velocity += self.acceleration * self.dt
        self.position += self.velocity * self.dt
        
        # ✅ Apply orientation updates
        self.orientation += self.angular_velocity * self.dt
        self.angular_velocity += self.angular_acceleration * self.dt

        # ✅ Normalize orientation angles
        self.orientation = np.mod(self.orientation + np.pi, 2 * np.pi) - np.pi

        # ✅ Debug info
        print(f"🔄 State Updated: Position {self.position}, Velocity {self.velocity}, Orientation {self.orientation}")

    def rotate_points(self, points, rotation):
        """
        Rotate world points (green dots) based on orientation.
        """
        roll, pitch, yaw = rotation

        # Roll rotation (X-axis)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch rotation (Y-axis)
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw rotation (Z-axis)
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combine rotations: Yaw → Pitch → Roll
        R = R_z @ R_y @ R_x

        # Apply rotation to all points
        rotated_points = R @ points

        return rotated_points

    def animate_scene(self, num_frames=200):
        """
        Animate the 3D scene with integrated motion from acceleration & orientation.
        """
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()

            # Debug print to track acceleration values
            print(f"Current acceleration: {self.acceleration}, magnitude: {np.linalg.norm(self.acceleration):.2f} m/s²")
            
            # Update dynamics
            self.propagate_dynamics_primary_object()

            # Move world points relative to integrated motion
            relative_position = -self.position.reshape(3, 1)

            # ✅ Rotate the world points based on orientation
            rotated_world_points = self.rotate_points(self.world_points + relative_position, self.orientation)

            # Plot the primary object
            ax.scatter(
                self.primary_object_point_cloud[0],
                self.primary_object_point_cloud[1],
                self.primary_object_point_cloud[2],
                c='b',
                alpha=0.3,
                s=1
            )

            # ✅ Plot rotated world points (GREEN DOTS)
            ax.scatter(
                rotated_world_points[0],
                rotated_world_points[1],
                rotated_world_points[2],
                c='g',
                alpha=0.6,
                s=5
            )
            
            # Plot the dynamic surrounding point cloud if available
            if self.surrounding_objects_point_cloud is not None and len(self.surrounding_objects_point_cloud) > 0:
                # Ensure points are in correct shape (3, N)
                if self.surrounding_objects_point_cloud.shape[1] == 3:
                    points = self.surrounding_objects_point_cloud.T
                else:
                    points = self.surrounding_objects_point_cloud
                    
                # Plot the surrounding points
                ax.scatter(
                    points[0], points[1], points[2],
                    c='r',  # Different color to distinguish from primary object
                    alpha=0.5,
                    s=2,
                    label='Surrounding Points'
                )
            
            # Plot velocity vector
            velocity_magnitude = np.linalg.norm(self.velocity)
            if velocity_magnitude > 0:
                normalized_velocity = self.velocity / velocity_magnitude
                ax.quiver(0, 0, 0,
                          normalized_velocity[0], normalized_velocity[1], normalized_velocity[2],
                          color='blue', alpha=0.8, length=velocity_magnitude)

            # Acceleration vector
            accel_magnitude = np.linalg.norm(self.acceleration)
            if accel_magnitude > 0:
                normalized_accel = self.acceleration / accel_magnitude
                ax.quiver(0, 0, 0,
                          normalized_accel[0], normalized_accel[1], normalized_accel[2],
                          color='red', alpha=0.8, length=accel_magnitude,
                          linestyle='dashed')

            # Set consistent view
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-10, 10])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Update title with verified acceleration value
            ax.set_title(
                f'Frame {frame}\n'
                f'Position: [{self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f}]\n'
                f'Velocity: {velocity_magnitude:.2f} m/s\n'
                f'Acceleration: {accel_magnitude:.2f} m/s²\n'
                f'Points in cloud: {len(self.surrounding_objects_point_cloud) if self.surrounding_objects_point_cloud is not None else 0}'
                f'Orientation: {self.orientation}'
            )
            
            ax.set_box_aspect([1,1,1])
            ax.legend()
            
            return tuple(ax.get_children())

        ani = animation.FuncAnimation(
            fig, 
            update, 
            frames=num_frames,
            interval=50,
            blit=False,
            repeat=True
        )

        plt.show()

    

    def set_point_cloud(self, points):
        """
        Set the point cloud data for visualization
        Args:
            points: numpy array of shape (N, 3) containing point cloud coordinates
        """
        self.point_cloud = points
        print(f"Point cloud set with {len(points)} points")
    
    def transform_points(self, points):
        """
        Transform point cloud based on current position and orientation
        Args:
            points: numpy array of shape (N, 3)
        Returns:
            transformed points: numpy array of shape (N, 3)
        """
        # Create rotation matrix from current orientation
        R = self.get_rotation_matrix()  # You'll need to implement this based on your orientation representation
        
        # Apply rotation and translation to all points
        transformed = np.dot(points, R.T) + self.position
        
        return transformed
    
    def get_rotation_matrix(self):
        """
        Get the current rotation matrix based on orientation
        Returns:
            R: 3x3 rotation matrix
        """
        # Implement based on how you represent orientation
        # This is a placeholder that returns identity matrix
        return np.eye(3)

    def update_surrounding_point_cloud(self, new_points: np.ndarray) -> None:
        """
        Updates the surrounding objects point cloud with new data
        Args:
            new_points: numpy array of shape (N, 3) containing new point cloud data
        """
        if new_points is not None and len(new_points) > 0:
            self.surrounding_objects_point_cloud = new_points
            print(f"\nUpdated surrounding point cloud with {len(new_points)} points")
            print(f"Sample of first 5 points:")
            print(new_points[:5] if len(new_points) >= 5 else new_points)
        else:
            print("Warning: Received empty point cloud data")
            
            
if __name__ == "__main__":
    print("🚀 Starting Simulation...")
    state = simState()
    
    # Create and start the update thread
    update_thread = threading.Thread(target=state.update_state, daemon=True)
    update_thread.start()
    
    # Start the animation
    state.animate_scene()
    
    print("✅ Simulation running!")
