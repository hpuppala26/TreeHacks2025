import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.image import imread

# 1. Map Initialization
map_img = imread('city_map.png')  # Georeferenced PNG
map_height, map_width, _ = map_img.shape
map_bounds = {'lat_min': 32.715, 'lon_min': -117.162,  # San Diego
              'lat_max': 32.755, 'lon_max': -117.122}

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(map_img, extent=[0, map_width, 0, map_height])
ax.set_axis_off()

# 2. Vehicle Art Assets
vehicle_icon, = ax.plot([], [], 'ro', ms=10, alpha=0.7)
path_line, = ax.plot([], [], 'b-', lw=2, alpha=0.5)
history_x, history_y = [], []

# 3. Animation Core
def frame_gen():
    while True:
        yield (np.random.uniform(map_bounds['lon_min'], map_bounds['lon_max']),
               np.random.uniform(map_bounds['lat_min'], map_bounds['lat_max']))

def update_frame(geo_pos):
    global history_x, history_y
    
    # Coordinate transformation
    x, y = geo_to_pixel(*geo_pos, map_bounds)
    
    # Update vehicle position
    vehicle_icon.set_data(x, y)
    
    # Update path history
    history_x.append(x)
    history_y.append(y)
    path_line.set_data(history_x[-100:], history_y[-100:])  # 100-point trail
    
    return vehicle_icon, path_line

# 4. Launch Animation
ani = FuncAnimation(fig, update_frame, frames=frame_gen,
                   interval=33, blit=True, cache_frame_data=False)
plt.show()
