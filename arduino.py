# import serial
# import matplotlib.pyplot as plt
# import time

# # Serial port configuration
# PORT = "/dev/tty.usbserial-1120"  # Replace with your port
# BAUDRATE = 115200

# ser = serial.Serial(PORT, BAUDRATE, timeout=1)

# # Data storage for plotting
# times, x_values, y_values, z_values = [], [], [], []

# # Enable interactive plotting
# plt.ion()
# fig, ax = plt.subplots()
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Acceleration (m/s²)")

# def update_plot():
#     """
#     Update the live plot with new accelerometer data.
#     """
#     ax.clear()
#     ax.plot(times, x_values, label="X-axis", color='red')
#     ax.plot(times, y_values, label="Y-axis", color='green')
#     ax.plot(times, z_values, label="Z-axis", color='blue')
#     ax.legend()
#     plt.pause(0.01)

# def convert_raw_to_acceleration(raw_value):
#     """
#     Convert raw accelerometer data to acceleration in m/s².
#     Assumes ±2g range with 0.004 g/LSB sensitivity.
#     """
#     sensitivity = 0.004  # g/LSB for ±2g range
#     g_to_ms2 = 9.81      # Conversion factor from g to m/s²

#     # Convert raw value to acceleration in g and then to m/s²
#     acceleration_ms2 = raw_value * sensitivity * g_to_ms2
#     return acceleration_ms2

# def read_accelerometer():
#     """
#     Read accelerometer data from the serial port and process it.
#     """
#     start_time = time.time()
    
#     while True:
#         if ser.in_waiting > 0:
#             # Read a line of data from the serial port
#             data = ser.readline().decode('utf-8', errors='ignore').strip()

#             # Parse the accelerometer data if it contains "X:"
#             if "X:" in data:
#                 try:
#                     # Extract raw acceleration values (in m/s²)
#                     raw_x, raw_y, raw_z = [int(v.split(":")[1].strip()) for v in data.split("|")]

#                     # Convert raw values to acceleration in m/s²
#                     x_accel = convert_raw_to_acceleration(raw_x)
#                     y_accel = convert_raw_to_acceleration(raw_y)
#                     z_accel = convert_raw_to_acceleration(raw_z)

#                     # Update time and acceleration data
#                     times.append(time.time() - start_time)
#                     x_values.append(x_accel)
#                     y_values.append(y_accel)
#                     z_values.append(z_accel)

#                     # Keep only the last 100 data points for plotting
#                     if len(times) > 100:
#                         times.pop(0)
#                         x_values.pop(0)
#                         y_values.pop(0)
#                         z_values.pop(0)

#                     # Update the live plot
#                     update_plot()

#                     # Print acceleration values to the console
#                     print(f"📡 X: {x_accel:.2f} m/s² | Y: {y_accel:.2f} m/s² | Z: {z_accel:.2f} m/s²")

#                 except ValueError:
#                     pass

# try:
#     read_accelerometer()
# except KeyboardInterrupt:
#     ser.close()
#     print("Serial Connection Closed.")


# import serial
# import numpy as np
# import matplotlib.pyplot as plt
# import time

# # Serial port configuration
# PORT = "/dev/tty.usbserial-120"  # Replace with your actual port
# BAUDRATE = 19200  # Match the baud rate in the Arduino code

# ser = serial.Serial(PORT, BAUDRATE, timeout=1)

# # Constants
# sensitivity = 9.81  # Since Arduino already scales by 16384.0, we use 9.81 directly
# max_points = 100     # Number of data points to keep for plotting

# # Data storage (NumPy for efficiency)
# times = np.linspace(-max_points, 0, max_points)  # Start with negative time for smooth scrolling
# x_values = np.zeros(max_points)
# y_values = np.zeros(max_points)
# z_values = np.zeros(max_points)

# # Matplotlib setup
# plt.ion()
# fig, ax = plt.subplots()
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Acceleration (m/s²)")
# ax.set_ylim(-10, 10)  # Adjust Y-axis limits for better visibility
# line_x, = ax.plot(times, x_values, label="X-axis", color='red')
# line_y, = ax.plot(times, y_values, label="Y-axis", color='green')
# line_z, = ax.plot(times, z_values, label="Z-axis", color='blue')
# ax.legend()

# start_time = time.time()  # Initialize time tracking


# def update_plot():
#     """
#     Efficiently updates the live plot without clearing and redrawing everything.
#     """
#     line_x.set_xdata(times)
#     line_y.set_xdata(times)
#     line_z.set_xdata(times)
    
#     line_x.set_ydata(x_values)
#     line_y.set_ydata(y_values)
#     line_z.set_ydata(z_values)
    
#     ax.set_xlim(times[0], times[-1])  # **Ensure X-axis updates correctly**
#     ax.relim()
#     ax.autoscale_view()
    
#     plt.draw()
#     plt.pause(0.01)  # Refresh rate of 10ms


# def read_accelerometer():
#     """
#     Reads accelerometer data from the serial port and processes it.
#     """
#     global times, x_values, y_values, z_values

#     accel_data = {"AccX": None, "AccY": None, "AccZ": None}
#     gyro_data = {"Roll": None, "Pitch": None, "Yaw": None}

#     while True:
#         if ser.in_waiting > 0:
#             # Read a line from the serial port
#             data = ser.read_until(b'\n').decode('utf-8', errors='ignore').strip()

#             # Process Accelerometer Data
#             if data.startswith("AccX"):
#                 try:
#                     key, value = data.split(":")
#                     accel_data[key.strip()] = float(value.strip()) * sensitivity
#                 except ValueError:
#                     continue  # Skip corrupt lines

#             elif data.startswith("AccY"):
#                 try:
#                     key, value = data.split(":")
#                     accel_data[key.strip()] = float(value.strip()) * sensitivity
#                 except ValueError:
#                     continue

#             elif data.startswith("AccZ"):
#                 try:
#                     key, value = data.split(":")
#                     accel_data[key.strip()] = float(value.strip()) * sensitivity
#                 except ValueError:
#                     continue

#             # Process Gyroscope Data
#             elif data.startswith("Roll"):
#                 try:
#                     key, value = data.split(":")
#                     gyro_data[key.strip()] = float(value.strip())
#                 except ValueError:
#                     continue

#             elif data.startswith("Pitch"):
#                 try:
#                     key, value = data.split(":")
#                     gyro_data[key.strip()] = float(value.strip())
#                 except ValueError:
#                     continue

#             elif data.startswith("Yaw"):
#                 try:
#                     key, value = data.split(":")
#                     gyro_data[key.strip()] = float(value.strip())
#                 except ValueError:
#                     continue

#             # If we have all accelerometer values, process and plot
#             if None not in accel_data.values():
#                 current_time = time.time() - start_time  # ✅ **Fix: Use Relative Time for X-axis**
                
#                 times[:-1] = times[1:]  # Shift time window
#                 times[-1] = current_time  # ✅ Properly updating time
                
#                 x_values[:-1] = x_values[1:]
#                 x_values[-1] = accel_data["AccX"]

#                 y_values[:-1] = y_values[1:]
#                 y_values[-1] = accel_data["AccY"]

#                 z_values[:-1] = z_values[1:]
#                 z_values[-1] = accel_data["AccZ"]

#                 # Update the live plot
#                 update_plot()

#                 # Print acceleration values to the console
#                 print(f"📡 X: {x_values[-1]:.2f} m/s² | Y: {y_values[-1]:.2f} m/s² | Z: {z_values[-1]:.2f} m/s²")

#                 # Reset the dictionary for the next batch of data
#                 accel_data = {"AccX": None, "AccY": None, "AccZ": None}

#             # Print gyroscope data when all values are received
#             if None not in gyro_data.values():
#                 print(f"🎯 Roll: {gyro_data['Roll']:.2f}° | Pitch: {gyro_data['Pitch']:.2f}° | Yaw: {gyro_data['Yaw']:.2f}°")
#                 gyro_data = {"Roll": None, "Pitch": None, "Yaw": None}


# try:
#     read_accelerometer()
# except KeyboardInterrupt:
#     ser.close()
#     print("Serial Connection Closed.")



import serial
import json
import time

# Serial port configuration
PORT = "/dev/tty.usbserial-120"  # Change this to your port
BAUDRATE = 19200
ser = serial.Serial(PORT, BAUDRATE, timeout=1)

# File to store sensor data
SENSOR_FILE = "sensor_data.json"

# Initialize default data
sensor_data = {
    "acceleration": {"AccX": 0, "AccY": 0, "AccZ": 0},
    "orientation": {"Roll": 0, "Pitch": 0, "Yaw": 0}
}

def write_sensor_data():
    """
    Continuously read sensor data from serial and write to a JSON file.
    """
    global sensor_data

    while True:
        if ser.in_waiting > 0:
            data = ser.read_until(b'\n').decode('utf-8', errors='ignore').strip()

            # Process Accelerometer Data
            if data.startswith("AccX"):
                try:
                    key, value = data.split(":")
                    sensor_data["acceleration"][key.strip()] = float(value.strip()) * 9.81
                except ValueError:
                    pass

            elif data.startswith("AccY"):
                try:
                    key, value = data.split(":")
                    sensor_data["acceleration"][key.strip()] = float(value.strip()) * 9.81
                except ValueError:
                    pass

            elif data.startswith("AccZ"):
                try:
                    key, value = data.split(":")
                    sensor_data["acceleration"][key.strip()] = float(value.strip()) * 9.81
                except ValueError:
                    pass

            # Process Gyroscope Data
            elif data.startswith("Roll"):
                try:
                    key, value = data.split(":")
                    sensor_data["orientation"][key.strip()] = float(value.strip())
                except ValueError:
                    pass

            elif data.startswith("Pitch"):
                try:
                    key, value = data.split(":")
                    sensor_data["orientation"][key.strip()] = float(value.strip())
                except ValueError:
                    pass

            elif data.startswith("Yaw"):
                try:
                    key, value = data.split(":")
                    sensor_data["orientation"][key.strip()] = float(value.strip())
                except ValueError:
                    pass

            # ✅ Write the latest data to a JSON file
            with open(SENSOR_FILE, "w") as f:
                json.dump(sensor_data, f, indent=4)

            # Print for debugging (optional)
            print(f"📡 Sensor Data Updated: {sensor_data}")

            # Small delay to avoid excessive CPU usage
            time.sleep(0.1)

# Start the logging process
if __name__ == '__main__':
    print("✅ Starting sensor logging... Press Ctrl+C to stop.")
    write_sensor_data()
