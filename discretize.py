import cv2
import numpy as np
import time
import os

class Discretizer:
    def __init__(self, rows=20, cols=40):
        self.rows = rows
        self.cols = cols
        self.edge_history = []
        self.depth_history = []
        self.timestamp_history = []
    
    def discretize_edge(self, edge_img, threshold=127):
        """Convert edge detection image to binary matrix"""
        resized = cv2.resize(edge_img, (self.cols, self.rows))
        return (resized > threshold).astype(int)
    
    def discretize_depth(self, depth_img, levels=5):
        """Convert depth map to matrix with specified number of levels"""
        resized = cv2.resize(depth_img, (self.cols, self.rows))
        # Normalize to 0-4 for 5 levels
        return (resized / (256/levels)).astype(int)
    
    def add_frame(self, edge_img, depth_img):
        """Process and store a new frame"""
        edge_matrix = self.discretize_edge(edge_img)
        depth_matrix = self.discretize_depth(depth_img)
        
        self.edge_history.append(edge_matrix)
        self.depth_history.append(depth_matrix)
        self.timestamp_history.append(time.time())
    
    def get_latest(self):
        """Get most recent discretized data"""
        if not self.edge_history:
            return None, None
        return self.edge_history[-1], self.depth_history[-1]
    
    def print_latest(self):
        """Print latest frame in terminal"""
        edge_matrix, depth_matrix = self.get_latest()
        if edge_matrix is None:
            return
        
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Print Edge Detection
        print("\nEdge Detection Matrix:")
        print(edge_matrix)
        
        # Print Depth Map
        print("\nDepth Map Matrix:")
        print(depth_matrix)
        
        # Visual representation (optional)
        print("\nVisual Edge Detection:")
        edge_chars = [' ', '█']
        for row in edge_matrix:
            print(''.join(edge_chars[val] for val in row))
        
        print("\nVisual Depth Map:")
        depth_chars = [' ', '░', '▒', '▓', '█']
        for row in depth_matrix:
            print(''.join(depth_chars[min(val, len(depth_chars)-1)] for val in row))
    
    def get_history(self):
        """Get all stored data"""
        return {
            'edge_history': self.edge_history,
            'depth_history': self.depth_history,
            'timestamps': self.timestamp_history
        }
    
    def clear_history(self):
        """Clear stored data"""
        self.edge_history = []
        self.depth_history = []
        self.timestamp_history = []