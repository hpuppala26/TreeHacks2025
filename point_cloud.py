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
        self.depth_points = None
        self.edge_points = None
        self.depth_fig = None
        self.edge_fig = None
        self.depth_ax = None
        self.edge_ax = None
        self.is_running = False
        plt.ion()  # Enable interactive mode
        
    def update_depth_cloud(self, depth_map):
        """Updates point cloud from depth data only"""
        try:
            points = self._generate_depth_points(depth_map)
            self.depth_points = points
            
            print("\nDepth Point Cloud Data:")
            print(f"Number of depth points: {len(points)}")
            if len(points) > 0:
                print("Sample of depth points (x, y, z):")
                print(points[:5])
                print("\nDepth point cloud statistics:")
                print(f"X range: {np.min(points[:, 0]):.3f} to {np.max(points[:, 0]):.3f}")
                print(f"Y range: {np.min(points[:, 1]):.3f} to {np.max(points[:, 1]):.3f}")
                print(f"Z range: {np.min(points[:, 2]):.3f} to {np.max(points[:, 2]):.3f}")
            
            self._update_depth_plot()
            return points
            
        except Exception as e:
            print(f"Error in update_depth_cloud: {e}")
            return None
            
    def update_edge_cloud(self, edge_map):
        """Updates point cloud from edge data only"""
        try:
            points = self._generate_edge_points(edge_map)
            self.edge_points = points
            
            print("\nEdge Point Cloud Data:")
            print(f"Number of edge points: {len(points)}")
            if len(points) > 0:
                print("Sample of edge points (x, y, z):")
                print(points[:5])
                print("\nEdge point cloud statistics:")
                print(f"X range: {np.min(points[:, 0]):.3f} to {np.max(points[:, 0]):.3f}")
                print(f"Y range: {np.min(points[:, 1]):.3f} to {np.max(points[:, 1]):.3f}")
                print(f"Z range: {np.min(points[:, 2]):.3f} to {np.max(points[:, 2]):.3f}")
            
            self._update_edge_plot()
            return points
            
        except Exception as e:
            print(f"Error in update_edge_cloud: {e}")
            return None
    
    def _generate_depth_points(self, depth_map):
        """Generates 3D points from depth map only"""
        points = []
        depth_normalized = depth_map.astype(float) / 255.0
        
        depth_threshold = 0.1
        step = 4
        
        for y in range(0, self.frame_height, step):
            for x in range(0, self.frame_width, step):
                if depth_normalized[y, x] > depth_threshold:
                    z = depth_normalized[y, x] * 5
                    x_norm = (x - self.frame_width/2) / (self.frame_width/2)
                    y_norm = (y - self.frame_height/2) / (self.frame_height/2)
                    points.append([x_norm, y_norm, z])
        
        return np.array(points) if points else np.zeros((1, 3))
    
    def _generate_edge_points(self, edge_map):
        """Generates 3D points from edge detection only"""
        points = []
        edge_threshold = 50
        step = 4
        
        for y in range(0, self.frame_height, step):
            for x in range(0, self.frame_width, step):
                if edge_map[y, x] > edge_threshold:
                    # Use normalized edge intensity for z-coordinate
                    z = edge_map[y, x] / 255.0
                    x_norm = (x - self.frame_width/2) / (self.frame_width/2)
                    y_norm = (y - self.frame_height/2) / (self.frame_height/2)
                    points.append([x_norm, y_norm, z])
        
        return np.array(points) if points else np.zeros((1, 3))
    
    def _update_depth_plot(self):
        """Updates the depth point cloud plot"""
        if self.depth_points is None or len(self.depth_points) == 0:
            return
            
        if self.depth_fig is None:
            self.depth_fig = plt.figure(figsize=(8, 6))
            self.depth_ax = self.depth_fig.add_subplot(111, projection='3d')
            plt.show(block=False)
        
        self.depth_ax.cla()
        self.depth_ax.scatter(self.depth_points[:, 0], 
                            self.depth_points[:, 1], 
                            self.depth_points[:, 2],
                            c='b', marker='o', s=1)
        
        self.depth_ax.set_xlabel('X')
        self.depth_ax.set_ylabel('Y')
        self.depth_ax.set_zlabel('Z')
        self.depth_ax.set_title(f'Depth Point Cloud ({len(self.depth_points)} points)')
        
        self.depth_fig.canvas.draw()
        self.depth_fig.canvas.flush_events()
    
    def _update_edge_plot(self):
        """Updates the edge point cloud plot"""
        if self.edge_points is None or len(self.edge_points) == 0:
            return
            
        if self.edge_fig is None:
            self.edge_fig = plt.figure(figsize=(8, 6))
            self.edge_ax = self.edge_fig.add_subplot(111, projection='3d')
            plt.show(block=False)
        
        self.edge_ax.cla()
        self.edge_ax.scatter(self.edge_points[:, 0], 
                           self.edge_points[:, 1], 
                           self.edge_points[:, 2],
                           c='r', marker='o', s=1)
        
        self.edge_ax.set_xlabel('X')
        self.edge_ax.set_ylabel('Y')
        self.edge_ax.set_zlabel('Z')
        self.edge_ax.set_title(f'Edge Point Cloud ({len(self.edge_points)} points)')
        
        self.edge_fig.canvas.draw()
        self.edge_fig.canvas.flush_events()
    
    def close(self):
        """Clean up visualization"""
        if self.depth_fig is not None:
            plt.close(self.depth_fig)
        if self.edge_fig is not None:
            plt.close(self.edge_fig)
        self.depth_fig = None
        self.edge_fig = None

    def display_voxel_grid(self, voxel_size=0.01):
        """Displays a voxel grid created from the point cloud"""
        if self.depth_points is None and self.edge_points is None:
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
        if self.depth_points is None and self.edge_points is None:
            return None
            
        # Create voxel grid
        points = np.concatenate([self.depth_points, self.edge_points]) if self.depth_points is not None and self.edge_points is not None else (self.depth_points if self.depth_points is not None else self.edge_points)
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
            print(f'Displaying 3D geometry for {self.depth_points.shape[0] if self.depth_points is not None else self.edge_points.shape[0]:,} points')                
            plt.figure()
            plt.title("Point Cloud")
            ax = plt.axes(projection='3d')
            ax.scatter(self.depth_points[:, 0] if self.depth_points is not None else self.edge_points[:, 0], 
                       self.depth_points[:, 1] if self.depth_points is not None else self.edge_points[:, 1], 
                       self.depth_points[:, 2] if self.depth_points is not None else self.edge_points[:, 2], c='r', marker='o', s=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        except Exception as e:
            print(f'Error displaying geometry: {e}')
 