import serial
import matplotlib.pyplot as plt
import time

# Serial port configuration
PORT = "/dev/tty.usbserial-1120"  # Replace with your port
BAUDRATE = 115200

ser = serial.Serial(PORT, BAUDRATE, timeout=1)

# Data storage for plotting
times, x_values, y_values, z_values = [], [], [], []

# Enable interactive plotting
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration (m/s²)")

def update_plot():
    """
    Update the live plot with new accelerometer data.
    """
    ax.clear()
    ax.plot(times, x_values, label="X-axis", color='red')
    ax.plot(times, y_values, label="Y-axis", color='green')
    ax.plot(times, z_values, label="Z-axis", color='blue')
    ax.legend()
    plt.pause(0.01)

def convert_raw_to_acceleration(raw_value):
    """
    Convert raw accelerometer data to acceleration in m/s².
    Assumes ±2g range with 0.004 g/LSB sensitivity.
    """
    sensitivity = 0.004  # g/LSB for ±2g range
    g_to_ms2 = 9.81      # Conversion factor from g to m/s²

    # Convert raw value to acceleration in g and then to m/s²
    acceleration_ms2 = raw_value * sensitivity * g_to_ms2
    return acceleration_ms2

def read_accelerometer():
    """
    Read accelerometer data from the serial port and process it.
    """
    start_time = time.time()
    
    while True:
        if ser.in_waiting > 0:
            # Read a line of data from the serial port
            data = ser.readline().decode('utf-8', errors='ignore').strip()

            # Parse the accelerometer data if it contains "X:"
            if "X:" in data:
                try:
                    # Extract raw acceleration values (in m/s²)
                    raw_x, raw_y, raw_z = [int(v.split(":")[1].strip()) for v in data.split("|")]

                    # Convert raw values to acceleration in m/s²
                    x_accel = convert_raw_to_acceleration(raw_x)
                    y_accel = convert_raw_to_acceleration(raw_y)
                    z_accel = convert_raw_to_acceleration(raw_z)

                    # Update time and acceleration data
                    times.append(time.time() - start_time)
                    x_values.append(x_accel)
                    y_values.append(y_accel)
                    z_values.append(z_accel)

                    # Keep only the last 100 data points for plotting
                    if len(times) > 100:
                        times.pop(0)
                        x_values.pop(0)
                        y_values.pop(0)
                        z_values.pop(0)

                    # Update the live plot
                    update_plot()

                    # Print acceleration values to the console
                    print(f"📡 X: {x_accel:.2f} m/s² | Y: {y_accel:.2f} m/s² | Z: {z_accel:.2f} m/s²")

                except ValueError:
                    pass

try:
    read_accelerometer()
except KeyboardInterrupt:
    ser.close()
    print("Serial Connection Closed.")
