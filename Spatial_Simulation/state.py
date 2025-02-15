import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import matplotlib.animation as animation


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
    surrounding_objects_point_cloud: np.ndarray
    
    # Center point of the primary object (in global coordinates)
    primary_center: np.ndarray = None
    
    
    
    def propagate_dynamics_primary_object(self):
        """Single step propagation for animation"""
        self.current_time += self.dt  # Changed from time_step to dt
        # Update position based on velocity
        self.position += self.velocity * self.dt
        # Could add more physics here (acceleration, etc.)
        return self.position, self.velocity

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
        Generate simplified convex hull from points
        """
        # Downsample points first
        downsampled_points = self.voxel_downsample(points, voxel_size=0.2)
        
        # Convert from (3, N) to (N, 3) for ConvexHull
        points_transformed = downsampled_points.T
        
        # Generate convex hull
        hull = ConvexHull(points_transformed)
        
        return points_transformed[hull.vertices], hull.simplices
    
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
        Animate the scene with the primary body centered and world moving around it
        """
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            
            # Update dynamics using single step
            self.propagate_dynamics_primary_object()
            
            # Move world points relative to velocity
            relative_motion = -self.velocity.reshape(3, 1)  # Reshape for broadcasting
            self.world_points += relative_motion * self.dt
            
            # Plot primary object (centered)
            vertices, faces = self.generate_hull(self.primary_object_point_cloud)
            
            # Plot the point cloud
            ax.scatter(
                self.primary_object_point_cloud[0],
                self.primary_object_point_cloud[1],
                self.primary_object_point_cloud[2],
                c='b',
                alpha=0.3,
                s=1
            )
            
            # Plot the convex hull
            ax.plot_trisurf(
                vertices[:,0], vertices[:,1], vertices[:,2],
                triangles=faces,
                alpha=0.3,
                color='r'
            )
            
            # Plot world points
            ax.scatter(
                self.world_points[0],
                self.world_points[1],
                self.world_points[2],
                c='g',
                alpha=0.6,
                s=5
            )
            
            # Plot velocity vector
            velocity_magnitude = np.linalg.norm(self.velocity)
            if velocity_magnitude > 0:
                normalized_velocity = self.velocity / velocity_magnitude
                ax.quiver(0, 0, 0,
                         normalized_velocity[0], normalized_velocity[1], normalized_velocity[2],
                         color='blue', alpha=0.8, length=velocity_magnitude)
            
            # Set consistent view
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-10, 10])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Frame {frame}, Velocity: {np.linalg.norm(self.velocity):.2f} m/s')
            
            # Ensure the aspect ratio is equal
            ax.set_box_aspect([1,1,1])
            
            # Return a tuple of artists
            return tuple(ax.get_children())
        
        ani = animation.FuncAnimation(
            fig, 
            update, 
            frames=num_frames,
            interval=50,  # 50ms between frames
            blit=False,  # Changed to False to avoid the _resize_id error
            repeat=True
        )
        
        plt.show()
    
    def __init__(self):
        # Create a sphere for the primary object
        radius = 1.0  # radius of 1 unit
        n_points = 1000  # number of points to represent the sphere
        
        # Generate spherical coordinates
        phi = np.random.uniform(0, 2*np.pi, n_points)
        theta = np.arccos(np.random.uniform(-1, 1, n_points))
        
        # Convert to Cartesian coordinates (3xN array)
        self.primary_object_point_cloud = np.array([
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta)
        ])
        
        # Initialize time parameters
        self.current_time = 0.0
        self.end_time = float('inf')  # For continuous animation
        self.dt = 0.1  # time step
        
        # Initialize physics parameters
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
        
        # Initialize world points
        n_world_points = 100
        self.world_points = np.random.uniform(-10, 10, (3, n_world_points))
        
        # Initialize other attributes
        self.surrounding_objects_point_cloud = np.array([])
        self.primary_center = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.angular_acceleration = np.zeros(3)
    
        