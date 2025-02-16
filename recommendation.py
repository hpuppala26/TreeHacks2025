import numpy as np

def get_best_direction(obstacles, plane_position, plane_heading):
    """
    Determines the best movement (Slide Left, Turn Left, etc.).
    Also ensures that the plane returns to its original straight path.
    """
    SAFE_DISTANCE = 15  
    TURN_DISTANCE = 30  

    left_clearance, right_clearance, up_clearance, down_clearance = 100, 100, 100, 100
    obstacle_detected = False

    for obj in obstacles:
        x, y, z, distance = obj["x"], obj["y"], obj["z"], obj["distance"]

        delta_x = x - plane_position[0]
        delta_y = y - plane_position[1]
        delta_z = z - plane_position[2]

        # If object is directly ahead
        if distance < SAFE_DISTANCE and -5 < delta_x < 5:
            print(f"🚨 Obstacle Ahead: {obj['object']} at distance {distance}m!")
            obstacle_detected = True

            # Check side clearance
            if delta_x < 0:
                left_clearance = min(left_clearance, distance)
            else:
                right_clearance = min(right_clearance, distance)

            # Check vertical clearance
            if delta_z > 0:
                up_clearance = min(up_clearance, distance)
            else:
                down_clearance = min(down_clearance, distance)

    # Decision Logic
    if obstacle_detected:
        if left_clearance > SAFE_DISTANCE:
            return "TURN LEFT" if left_clearance > TURN_DISTANCE else "SLIDE LEFT"
        elif right_clearance > SAFE_DISTANCE:
            return "TURN RIGHT" if right_clearance > TURN_DISTANCE else "SLIDE RIGHT"
        elif up_clearance > SAFE_DISTANCE:
            return "UP"
        elif down_clearance > SAFE_DISTANCE:
            return "DOWN"
        else:
            return "STOP"  # Emergency stop if no space

    # 🚀 If no obstacles detected, return to STRAIGHT flight
    if plane_heading != 0:
        return "RETURN TO STRAIGHT"
    
    return "STRAIGHT"

# Example obstacle data
obstacles = [
    {"object": "building", "x": 5, "y": 20, "z": 10, "distance": 10},
    {"object": "drone", "x": -10, "y": 15, "z": 5, "distance": 12}
]

plane_position = (0, 10, 5)  # (x, y, z) coordinates of plane
plane_heading = 10  # Assume the plane turned slightly
direction = get_best_direction(obstacles, plane_position, plane_heading)
print(f"🛫 Recommended Movement: {direction}")
