import numpy as np
import time

def check_immediate_collision(primary_object_point_cloud, surrounding_objects_point_cloud, safety_radius=5):
    for primary_point in primary_object_point_cloud:
        for obstacle_point in surrounding_objects_point_cloud:
            if np.linalg.norm(primary_point - obstacle_point) < safety_radius:
                print("🚨 ALERT! Immediate Collision Detected! 🚨")
                return True

    return False


def predict_future_collision(primary_object_point_cloud, primary_velocity,
                             surrounding_objects_point_cloud, time_horizon=10, time_step=1, safety_radius=5):
    for t in np.arange(1, time_horizon + 1, time_step):  # Check every second
        # Predict future airplane point cloud positions
        future_plane_point_cloud = primary_object_point_cloud + (primary_velocity * t)

        # Check if any future obstacle point collides with the plane
        for future_plane_point in future_plane_point_cloud:
            for obstacle_point in surrounding_objects_point_cloud:
                if np.linalg.norm(future_plane_point - obstacle_point) < safety_radius:
                    print(f"🚨 WARNING! Collision predicted in {t} seconds! 🚨")
                    
                    # Generate recommended action based on position differences
                    if obstacle_point[2] < future_plane_point[2]:  # If object is below
                        return "Climb Up 20m"
                    elif obstacle_point[0] > future_plane_point[0]:  # If object is to the right
                        return "Turn Left 30°"
                    else:
                        return "Reduce Speed & Change Course"

    return "No risk detected"


def real_time_collision_detection(primary_object_point_cloud, primary_velocity,
                                  surrounding_objects_point_cloud, time_horizon=10, time_step=1, safety_radius=5):
    for t in range(1, time_horizon + 1, time_step):
        print(f"\n⏳ Checking at time t = {t} seconds...")

        # Update positions dynamically based on velocity
        primary_object_point_cloud += primary_velocity * time_step

        # Check for immediate collision
        if check_immediate_collision(primary_object_point_cloud, surrounding_objects_point_cloud, safety_radius):
            print("🚨 ALERT! Immediate Collision Detected! 🚨")
            return

        # Check for predicted collision
        action = predict_future_collision(primary_object_point_cloud, primary_velocity,
                                          surrounding_objects_point_cloud, time_horizon, time_step, safety_radius)

        print("Recommended Action:", action)
        time.sleep(time_step)  # Simulate real-time execution


# Example 3D point cloud of airplane (N points)
primary_object_point_cloud = np.array([
    [50.1, 100.3, 500.0], 
    [50.5, 100.6, 500.2], 
    [51.0, 101.0, 500.5] 
])

# Example 3D point cloud of surrounding objects (M points)
surrounding_objects_point_cloud = np.array([
    [55.0, 105.0, 500.0],  
    [60.2, 110.3, 505.5],  
    [50.2, 100.5, 500.1]  
])

# Velocity of airplane (vx, vy, vz)
primary_velocity = np.array([5, -2, 3])  # Moving in x, y, z directions

# Run real-time detection
real_time_collision_detection(primary_object_point_cloud, primary_velocity, surrounding_objects_point_cloud)
