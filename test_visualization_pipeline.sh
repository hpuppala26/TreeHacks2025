#!/bin/bash

# Start Flask server (app.py) in the background
echo "Starting Flask server..."
python app.py &
APP_PID=$!

# Wait for Flask server to initialize
sleep 2

# Start realtime video processing in the background
echo "Starting realtime video processing..."
python realtime_video.py &
VIDEO_PID=$!

# Wait for video processing to initialize
echo "Waiting for video processing to stabilize..."
sleep 3

# Start visualization
echo "Starting visualization..."
python Spatial_Simulation/test_visualization.py

# Cleanup: Kill background processes when visualization ends
echo "Cleaning up processes..."
kill $APP_PID
kill $VIDEO_PID

echo "Pipeline completed."

#run using ./test_visualization_pipeline.sh