import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import matplotlib.animation as animation
import json
import time


# Define State Classs
class simState:
    
    # Define Time Variables:
    start_time: float
    time_step: float
    current_time: float
    end_time: float
    
    # Define State Variables:
    # Shape: (3, N) arrays where rows represent x,y,z components
    position: np.ndarray    # type: np.ndarray[3, N]
    velocity: np.ndarray    # type: np.ndarray[3, N]
    acceleration: np.ndarray # type: np.ndarray[3, N]
    orientation: np.ndarray # type: np.ndarray[3, N]
    angular_velocity: np.ndarray # type: np.ndarray[3, N]
    angular_acceleration: np.ndarray # type: np.ndarray[3, N]
    
    
    # Define an array that defines the point cloud of the primary object:
    primary_object_point_cloud: np.ndarray
    
    # Define an array that defines the point cloud of all surrounding objects:
    # write to this for the point cloud
    surrounding_objects_point_cloud: np.ndarray
    
    # Center point of the primary object (in global coordinates)
    primary_center: np.ndarray = None
    
    point_cloud = None  # Add this line to store point cloud data
    
    def integrate_acceleration(self):
        """
        Integrate acceleration to get velocity in local coordinates
        Uses trapezoidal integration for better accuracy
        
        Current acceleration is stored in self.acceleration (m/s²)
        Updates self.velocity (m/s)
        """
        # Previous velocity + (average acceleration × time step)
        self.velocity += self.acceleration * self.dt
        
        # Optional: Add damping to prevent unbounded velocity growth
        damping = 0.99  # Slight damping factor
        self.velocity *= damping
        
        return self.velocity
    
    def integrate_velocity(self):
        """
        Integrate velocity to get position in local coordinates
        Uses trapezoidal integration
        
        Current velocity is stored in self.velocity (m/s)
        Updates self.position (m)
        """
        # Previous position + (velocity × time step)
        self.position += self.velocity * self.dt
        
        return self.position
    
    def propagate_dynamics_primary_object(self):
        """
        Propagate the dynamics of the primary object using current acceleration data
        """
        self.current_time += self.dt
        
        # First integration: acceleration → velocity
        self.integrate_acceleration()
        
        # Second integration: velocity → position
        self.integrate_velocity()
        
        # Update angular motion (existing code)
        self.angular_velocity += self.angular_acceleration * self.dt
        self.orientation += self.angular_velocity * self.dt
        
        # Normalize orientation angles to [-π, π]
        self.orientation = np.mod(self.orientation + np.pi, 2 * np.pi) - np.pi
        
        return self.position, self.velocity, self.orientation, self.angular_velocity

    def update_primary_center(self) -> None:
        
        """Updates the center point of the primary object based on mean of its point cloud"""
        self.primary_center = np.mean(self.primary_object_point_cloud, axis=0)
    
    def to_local_coordinates(self, points: np.ndarray) -> np.ndarray:
        """
        Convert points from global to local coordinate system.
        Local system is centered at the primary object's center.
        
        Args:
            points: Array of shape (N, 3) containing points in global coordinates
            
        Returns:
            Array of shape (N, 3) containing points in local coordinates
        """
        if self.primary_center is None:
            self.update_primary_center()
        return points - self.primary_center
    
    def to_global_coordinates(self, points: np.ndarray) -> np.ndarray:
        """
        Convert points from local back to global coordinate system
        
        Args:
            points: Array of shape (N, 3) containing points in local coordinates
            
        Returns:
            Array of shape (N, 3) containing points in global coordinates
        """
        if self.primary_center is None:
            self.update_primary_center()
        return points + self.primary_center
    
    def rotate_points(self, points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """
        Rotate points based on pitch, roll, yaw angles.
        
        Args:
            points: Array of shape (3, N) containing points in local coordinates
            rotation: Array of shape (3,) containing [roll, pitch, yaw] in radians
            
        Returns:
            Array of shape (3, N) containing rotated points
        """
        roll, pitch, yaw = rotation
        
        # Roll rotation (around x-axis)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch rotation (around y-axis)
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw rotation (around z-axis)
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (order: yaw -> pitch -> roll)
        R = R_x @ R_y @ R_z
        
        # Apply rotation to all points
        rotated_points = R @ points
        
        return rotated_points
    
    def voxel_downsample(self, points: np.ndarray, voxel_size: float = 0.2) -> np.ndarray:
        """
        Downsample points using a voxel grid filter
        
        Args:
            points: Array of shape (3, N) containing points
            voxel_size: Size of voxel for downsampling (larger = fewer points)
            
        Returns:
            Downsampled points array of shape (3, M) where M < N
        """
        # Transform to (N, 3) for easier processing
        points_transformed = points.T
        
        # Compute voxel indices for each point
        voxel_indices = np.floor(points_transformed / voxel_size)
        
        # Dictionary to store voxel centers
        voxel_centers = {}
        
        # For each point, add to corresponding voxel
        for i, index in enumerate(voxel_indices):
            index_tuple = tuple(index)
            if index_tuple in voxel_centers:
                voxel_centers[index_tuple].append(points_transformed[i])
            else:
                voxel_centers[index_tuple] = [points_transformed[i]]
        
        # Compute mean point for each voxel
        downsampled_points = np.array([np.mean(points, axis=0) 
                                     for points in voxel_centers.values()])
        
        # Return in original (3, N) format
        return downsampled_points.T

    def generate_hull(self, points: np.ndarray):
        """
        Generate convex hull from points
        
        Args:
            points: Array of shape (3, N) containing points
            
        Returns:
            vertices: Array of hull vertices
            faces: Array of face indices
        """
        # Convert from (3, N) to (N, 3) for ConvexHull
        points_transformed = points.T
        
        # Generate convex hull
        hull = ConvexHull(points_transformed)
        
        # Get vertices in correct order
        vertices = points_transformed[hull.vertices]
        
        # Ensure faces indices are within bounds
        faces = hull.simplices
        if np.max(faces) >= len(vertices):
            # Reindex faces to match vertices
            faces = faces % len(vertices)
        
        return vertices, faces

    def visualize_primary_object(self):
        """
        Visualize the primary object point cloud and its convex hull in 3D
        """
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get points in correct format for hull generation
        points = self.primary_object_point_cloud
        
        # Generate hull
        vertices, faces = self.generate_hull(points)
        
        # Plot the point cloud
        ax.scatter(
            points[0], points[1], points[2],
            c='b',
            alpha=0.3,
            s=1,
            label='Point Cloud'
        )
        
        # Plot the convex hull
        ax.plot_trisurf(
            vertices[:,0], vertices[:,1], vertices[:,2],
            triangles=faces,
            alpha=0.3,
            color='r',
            label='Convex Hull'
        )
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Primary Object: Point Cloud and Convex Hull')
        
        # Add a grid
        ax.grid(True)
        
        # Add legend
        ax.legend()
        
        plt.show()
    
    def animate_scene(self, num_frames=200):
        """
        Animate the scene with integrated motion from acceleration data
        """
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            
            # Update dynamics using acceleration integration
            self.propagate_dynamics_primary_object()
            
            # Move world points relative to integrated motion
            relative_position = -self.position.reshape(3, 1)
            relative_rotation = -self.orientation
            
            # First translate then rotate world points
            translated_points = self.world_points + relative_position
            rotated_world_points = self.rotate_points(translated_points, relative_rotation)
            
            # Plot the fixed primary object - swap X and Z when plotting
            vertices, faces = self.generate_hull(self.primary_object_point_cloud)
            
            # Plot the point cloud (fixed) - swap X and Z coordinates
            ax.scatter(
                self.primary_object_point_cloud[2],  # Was [0]
                self.primary_object_point_cloud[1],  # Stays [1]
                self.primary_object_point_cloud[0],  # Was [2]
                c='b',
                alpha=0.3,
                s=1
            )
            
            # Plot the convex hull - swap X and Z coordinates
            ax.plot_trisurf(
                vertices[:,2],  # Was [:,0]
                vertices[:,1],  # Stays [:,1]
                vertices[:,0],  # Was [:,2]
                triangles=faces,
                alpha=0.3,
                color='r'
            )
            
            # Plot transformed world points - swap X and Z coordinates
            ax.scatter(
                rotated_world_points[2],  # Was [0]
                rotated_world_points[1],  # Stays [1]
                rotated_world_points[0],  # Was [2]
                c='g',
                alpha=0.6,
                s=5
            )
            
            # Plot the dynamic surrounding point cloud if available - swap X and Z coordinates
            if self.surrounding_objects_point_cloud is not None and len(self.surrounding_objects_point_cloud) > 0:
                if self.surrounding_objects_point_cloud.shape[1] == 3:
                    points = self.surrounding_objects_point_cloud.T
                else:
                    points = self.surrounding_objects_point_cloud
                    
                ax.scatter(
                    points[2],  # Was [0]
                    points[1],  # Stays [1]
                    points[0],  # Was [2]
                    c='r',
                    alpha=0.5,
                    s=2,
                    label='Surrounding Points'
                )
            
            # Plot velocity vector - swap X and Z components
            velocity_magnitude = np.linalg.norm(self.velocity)
            if velocity_magnitude > 0:
                normalized_velocity = self.velocity / velocity_magnitude
                ax.quiver(0, 0, 0,
                         normalized_velocity[2],  # Was [0]
                         normalized_velocity[1],  # Stays [1]
                         normalized_velocity[0],  # Was [2]
                         color='blue', alpha=0.8, length=velocity_magnitude)
            
            # Plot acceleration vector - swap X and Z components
            accel_magnitude = np.linalg.norm(self.acceleration)
            if accel_magnitude > 0:
                normalized_accel = self.acceleration / accel_magnitude
                ax.quiver(0, 0, 0,
                         normalized_accel[2],  # Was [0]
                         normalized_accel[1],  # Stays [1]
                         normalized_accel[0],  # Was [2]
                         color='red', alpha=0.8, length=accel_magnitude,
                         linestyle='dashed')
            
            # Set consistent view
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-10, 10])
            ax.set_xlabel('Z')
            ax.set_ylabel('Y')
            ax.set_zlabel('X')
            
            # Update title with swapped position components
            ax.set_title(
                f'Frame {frame}\n'
                f'Position: [{self.position[2]:.1f}, {self.position[1]:.1f}, {self.position[0]:.1f}]\n'
                f'Velocity: {velocity_magnitude:.2f} m/s\n'
                f'Acceleration: {accel_magnitude:.2f} m/s²\n'
                f'Points in cloud: {len(self.surrounding_objects_point_cloud) if self.surrounding_objects_point_cloud is not None else 0}'
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

    def read_sensor_data(self):
        """
        Reads the latest sensor data from the JSON file.
        """
        try:
            with open("/Users/hrithikpuppala/Desktop/treehacks-2025/sensor_data.json", "r") as f:
                sensor_data = json.load(f)

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
            print(f"⚠️ Warning: Sensor file not found or corrupted, using default values")
            return np.array([0.1, 0.0, 0.0]), np.zeros(3)  # Default values if file is missing

    def update_state(self):
        """
        Continuously updates the state with the latest sensor data.
        """
        while True:
            # ✅ Fetch latest sensor values
            self.acceleration, self.orientation = self.read_sensor_data()

            # ✅ Print updated values
            print(f"📡 UPDATED Acceleration: {self.acceleration}")
            print(f"📡 UPDATED Orientation: {self.orientation}")

            # ✅ Apply dynamics update
            self.propagate_dynamics_primary_object()

            # ✅ Sleep before next update (adjustable)
            time.sleep(0.5)  # Update every 0.5 seconds

    def propagate_dynamics_primary_object(self):
        """
        Updates state variables using acceleration data in real-time.
        """
        self.current_time += self.dt

        # ✅ Apply real-time acceleration updates
        self.velocity += self.acceleration * self.dt
        self.position += self.velocity * self.dt

        # ✅ Apply real-time orientation updates
        self.orientation = np.mod(self.orientation + np.pi, 2 * np.pi) - np.pi

        print(f"🔄 State Updated: Position {self.position}, Velocity {self.velocity}, Orientation {self.orientation}")

    
    def __init__(self):
        # Create a sphere for the primary object
        radius = 1.0  # radius of 1 unit
        n_points = 1000  # number of points to represent the sphere
        
        # Generate spherical coordinates
        phi = np.random.uniform(0, 2*np.pi, n_points)
        theta = np.arccos(np.random.uniform(-1, 1, n_points))
        
        # Convert to Cartesian coordinates (3xN array) - swap X and Z components
        self.primary_object_point_cloud = np.array([
            radius * np.cos(theta),  # This was X, now Z
            radius * np.sin(theta) * np.sin(phi),  # Y stays the same
            radius * np.sin(theta) * np.cos(phi)   # This was Z, now X
        ])
        
        # Initialize time parameters
        self.current_time = 0.0
        self.end_time = float('inf')  # For continuous animation
        self.dt = 0.1  # time step
        
        self.acceleration, self.orientation = self.read_sensor_data()
        
        # Initialize physics parameters
        self.velocity = np.zeros(3)
        # self.acceleration = np.array([0.1, 0.0, 0.0])
        self.position = np.zeros(3)
        # self.orientation = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.angular_acceleration = np.zeros(3)
        
        # Initialize world points
        n_world_points = 100
        self.world_points = np.random.uniform(-10, 10, (3, n_world_points))
        
        # Initialize other attributes
        self.surrounding_objects_point_cloud = np.array([])
        self.primary_center = np.zeros(3)

        # Load test point cloud data
        try:
            with open('test_point_cloud.json', 'r') as f:
                data = json.load(f)
                self.surrounding_objects_point_cloud = np.array(data['point_cloud'])
                print("\nLoaded test point cloud data:")
                print(f"Shape: {self.surrounding_objects_point_cloud.shape}")
                print("\nLast 5 entries:")
                print("-" * 50)
                for i, point in enumerate(self.surrounding_objects_point_cloud[-5:], 1):
                    print(f"Point {len(self.surrounding_objects_point_cloud)-5+i}: {point}")
                print("-" * 50)
        except Exception as e:
            print(f"Error loading test point cloud: {e}")
            self.surrounding_objects_point_cloud = np.array([])
        
        self.point_cloud = None  # Add this line to store point cloud data

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
    state = simState()
    state.update_state()
        