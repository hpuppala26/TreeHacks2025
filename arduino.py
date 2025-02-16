# import serial

# PORT = "/dev/tty.usbserial-120"  # Use the correct port from `ls /dev/tty.*`
# BAUDRATE = 115200

# ser = serial.Serial(PORT, BAUDRATE, timeout=1)

# def read_accelerometer():
#     while True:
#         if ser.in_waiting > 0:
#             data = ser.readline().decode('utf-8', errors='ignore').strip()
#             if "X:" in data:
#                 print(f"📡 Accelerometer Data: {data}")

# try:
#     read_accelerometer()
# except KeyboardInterrupt:
#     ser.close()
#     print("Serial Connection Closed.")





import serial
import matplotlib.pyplot as plt
import time

PORT = "/dev/tty.usbserial-120"
BAUDRATE = 115200

ser = serial.Serial(PORT, BAUDRATE, timeout=1)

times, x_values, y_values, z_values = [], [], [], []

plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration (m/s²)")

def update_plot():
    ax.clear()
    ax.plot(times, x_values, label="X-axis", color='red')
    ax.plot(times, y_values, label="Y-axis", color='green')
    ax.plot(times, z_values, label="Z-axis", color='blue')
    ax.legend()
    plt.pause(0.01)

def read_accelerometer():
    start_time = time.time()
    
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8', errors='ignore').strip()
            if "X:" in data:
                try:
                    x, y, z = [float(v.split(":")[1].split(" ")[1]) for v in data.split("|")]
                    times.append(time.time() - start_time)
                    x_values.append(x)
                    y_values.append(y)
                    z_values.append(z)

                    if len(times) > 100:
                        times.pop(0)
                        x_values.pop(0)
                        y_values.pop(0)
                        z_values.pop(0)

                    update_plot()
                    print(f"📡 X: {x:.2f} m/s² | Y: {y:.2f} m/s² | Z: {z:.2f} m/s²")

                except ValueError:
                    pass

try:
    read_accelerometer()
except KeyboardInterrupt:
    ser.close()
    print("Serial Connection Closed.")













# import numpy as np
# from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# # Simulated obstacle points detected by sensors (LiDAR, Camera, Radar)
# obstacle_points = np.array([
#     [10, 20, 5],  [15, 25, 8],  [20, 15, 6],  # Nearby objects
#     [40, 50, 10], [45, 55, 12], [35, 45, 9],  # Mid-range objects
#     [70, 80, 20], [75, 85, 22], [65, 75, 18], # Farther objects
# ])

# # Compute Convex Hull around the obstacles
# hull = ConvexHull(obstacle_points)

# # Plot 3D Convex Hull
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], obstacle_points[:, 2], color='blue')

# # Draw the Convex Hull
# for simplex in hull.simplices:
#     simplex = np.append(simplex, simplex[0])  # Close the loop
#     ax.plot(obstacle_points[simplex, 0], obstacle_points[simplex, 1], obstacle_points[simplex, 2], "r-")

# # ax.set_xlabel("X")
# # ax.set_ylabel("Y")
# # ax.set_zlabel("Z")
# # ax.set_title("3D Convex Hull of Obstacles (Safe Flight Path Calculation)")
# # plt.show()

# def get_realistic_flight_path(plane_position, convex_hull):
#     """
#     Determines the best realistic flight movement based on the convex hull.
#     The plane CANNOT move backward—it will turn instead.
#     """
#     hull_center = np.mean(convex_hull.points, axis=0)  # Center of safe zone

#     # Compute movement direction
#     direction_vector = hull_center - plane_position
#     direction_vector /= np.linalg.norm(direction_vector)  # Normalize

#     move_x = "Right" if direction_vector[0] > 0 else "Left"
#     move_z = "Up" if direction_vector[2] > 0 else "Down"

#     # 🔥 Fix: Planes **never move backward** → Instead, they turn!
#     if direction_vector[1] < 0:  # If it was suggesting "Backward"
#         move_y = "Turn Left" if direction_vector[0] > 0 else "Turn Right"
#     else:
#         move_y = "Forward"

#     return f"Move {move_x}, {move_y}, {move_z}"

# # Example: Plane is at (50, 50, 15)
# plane_position = np.array([50, 50, 15])


# import time
# plane_position = np.array([50, 50, 15])

# while True:
#     # Assume new obstacle positions are detected dynamically
#     obstacle_points += np.random.randint(-3, 3, size=obstacle_points.shape)

#     # Recompute Convex Hull
#     hull = ConvexHull(obstacle_points)

#     # Compute new recommended path
#     recommendation = get_realistic_flight_path(plane_position, hull)
#     print(f"🚀 Updated Recommendation: {recommendation}")

#     time.sleep(1)  # Update every second



