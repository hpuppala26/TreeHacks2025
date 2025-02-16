#!/bin/bash

# Start Flask server (app.py) in the background
echo "Starting Flask server..."
python3 app.py &
APP_PID=$!

# Wait for Flask server to initialize
sleep 5

# Start realtime video processing in the background
echo "Starting realtime video processing..."
python3 realtime_video.py &
VIDEO_PID=$!

# Wait for video processing to initialize
echo "Waiting for video processing to stabilize..."
sleep 6

# Start visualization
echo "Starting visualization..."
python3 Spatial_Simulation/test_visualization.py

# Cleanup: Kill background processes when visualization ends
echo "Cleaning up processes..."
kill $APP_PID
kill $VIDEO_PID

echo "Pipeline completed."

#run using ./test_visualization_pipeline.sh