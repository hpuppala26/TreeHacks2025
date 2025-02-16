import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
from matplotlib.animation import FuncAnimation

# Class for rendering point clouds from depth data
class CloudRenderer:
    def __init__(self, frame_width=480, frame_height=360):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.points = None
        self.fig = None
        self.ax = None
        self.is_running = False
        plt.ion()  # Enable interactive mode
        
    def update_cloud(self, depth_map, edge_map):
        """Updates point cloud with new depth and edge data"""
        try:
            # Generate points from depth and edge data
            points = self._generate_points(depth_map, edge_map)
            self.points = points
            
            # Print point cloud data
            print("\nPoint Cloud Data:")
            print(f"Number of points: {len(points)}")
            if len(points) > 0:
                sample_size = min(5, len(points))
                print("Sample of points (x, y, z):")
                print(points[:sample_size])
                print("\nPoint cloud statistics:")
                print(f"X range: {np.min(points[:, 0]):.3f} to {np.max(points[:, 0]):.3f}")
                print(f"Y range: {np.min(points[:, 1]):.3f} to {np.max(points[:, 1]):.3f}")
                print(f"Z range: {np.min(points[:, 2]):.3f} to {np.max(points[:, 2]):.3f}")
            
            # Initialize or update visualization
            self._update_plot()
            
            return points
            
        except Exception as e:
            print(f"Error in update_cloud: {e}")
            return None
    
    def _generate_points(self, depth_map, edge_map):
        """Generates 3D points from depth map and edge detection"""
        points = []
        depth_normalized = depth_map.astype(float) / 255.0
        
        # Print input data statistics
        print("\nInput Data Statistics:")
        print(f"Depth map range: {np.min(depth_normalized):.3f} to {np.max(depth_normalized):.3f}")
        print(f"Edge map range: {np.min(edge_map)} to {np.max(edge_map)}")
        
        # Threshold for considering a point
        edge_threshold = 50
        depth_threshold = 0.1
        
        # Sample points more sparsely for better performance
        step = 4  # Sample every 4th pixel
        for y in range(0, self.frame_height, step):
            for x in range(0, self.frame_width, step):
                if edge_map[y, x] > edge_threshold or depth_normalized[y, x] > depth_threshold:
                    z = depth_normalized[y, x] * 5
                    x_norm = (x - self.frame_width/2) / (self.frame_width/2)
                    y_norm = (y - self.frame_height/2) / (self.frame_height/2)
                    points.append([x_norm, y_norm, z])
        
        return np.array(points) if points else np.zeros((1, 3))
    
    def _update_plot(self):
        """Updates the matplotlib 3D scatter plot"""
        if self.points is None or len(self.points) == 0:
            return
            
        if self.fig is None:
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.scatter = self.ax.scatter([], [], [], c='r', marker='o', s=1)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            plt.show(block=False)
        
        # Clear previous points
        self.ax.cla()
        
        # Plot new points
        self.ax.scatter(self.points[:, 0], 
                       self.points[:, 1], 
                       self.points[:, 2],
                       c='r', marker='o', s=1)
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Point Cloud ({len(self.points)} points)')
        
        # Update plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Clean up visualization"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def display_voxel_grid(self, voxel_size=0.01):
        """Displays a voxel grid created from the point cloud"""
        if self.points is None:
            return
            
        # Create voxel grid
        voxel_grid = self._create_voxel_grid(voxel_size)
        
        plt.figure()
        plt.title("Voxel Grid")
        plt.imshow(voxel_grid)
        plt.colorbar()
        plt.show()

    # Displays a voxel grid created from the point cloud
    def _create_voxel_grid(self, voxel_size=0.01):
        if self.points is None:
            return None
            
        # Create voxel grid
        points = self.points
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        grid_shape = tuple((max_bound - min_bound) / voxel_size).astype(int) + 1
        voxel_grid = np.zeros(grid_shape, dtype=bool)
        
        for point in points:
            x = int((point[0] - min_bound[0]) / voxel_size)
            y = int((point[1] - min_bound[1]) / voxel_size)
            z = int((point[2] - min_bound[2]) / voxel_size)
            voxel_grid[z, y, x] = True
        
        return voxel_grid

    # Generic method to display any 3D geometry
    def _display_geometry(self, geometry):
        try: 
            print(f'Displaying 3D geometry for {self.points.shape[0]:,} points')                
            plt.figure()
            plt.title("Point Cloud")
            ax = plt.axes(projection='3d')
            ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='r', marker='o', s=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        except Exception as e:
            print(f'Error displaying geometry: {e}')
 