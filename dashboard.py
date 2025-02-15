import pygame
import socketio
import random
import time

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Realistic Airplane Obstacle Avoidance")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 100, 255)

# Load Airplane Image
plane_img = pygame.image.load("airplane.jpg")  # Load airplane image
plane_img = pygame.transform.scale(plane_img, (60, 60))  # Resize to fit

# Airplane position
plane_x, plane_y = WIDTH // 2, HEIGHT - 100  # Start at bottom-center
plane_speed_x = 2  # 🛑 Reduced side movement speed (was 3)
plane_speed_y = 2  # Forward (upward) speed
turning_speed = 2  # 🛑 Less sharp turns (was 3)

obstacles = []
avoiding = False
avoid_direction = None  # Stores the current avoidance direction
AVOID_DURATION = 20  # 🛑 Reduced avoidance time (was 50)
avoid_timer = 0  # Tracks how long the plane should avoid

# WebSocket client
sio = socketio.Client()

@sio.on("connect")
def on_connect():
    print("Connected to Flask server!")

@sio.on("obstacle_data")
def receive_data(data):
    global obstacles
    print("Received obstacle data:", data)  # Debugging

    # Store obstacle positions
    obstacles.clear()
    for obj in data["objects"]:
        obstacle_x = obj["bbox"][0] + (obj["bbox"][2] - obj["bbox"][0]) // 2  # Center of object
        obstacle_y = obj["bbox"][1] + (obj["bbox"][3] - obj["bbox"][1]) // 2  # Center of object
        obstacles.append({"x": obstacle_x, "y": obstacle_y, "label": obj["object"]})

sio.connect("http://127.0.0.1:5000")

def draw():
    screen.fill(WHITE)

    # Draw airplane image
    screen.blit(plane_img, (plane_x - 30, plane_y - 30))  # Center the plane

    # Draw obstacles
    for obstacle in obstacles:
        pygame.draw.circle(screen, RED, (obstacle["x"], obstacle["y"]), 20)
        font = pygame.font.Font(None, 24)
        text = font.render(obstacle["label"], True, (0, 0, 0))
        screen.blit(text, (obstacle["x"] - 10, obstacle["y"] - 30))

def avoid_collision():
    global plane_x, plane_y, avoiding, avoid_direction, avoid_timer

    plane_y -= plane_speed_y  # Move the plane upward continuously

    # Reset plane to bottom when it moves off-screen
    if plane_y < 0:
        plane_y = HEIGHT - 100
        plane_x = WIDTH // 2

    # If the plane is already avoiding an obstacle, continue in the chosen direction
    if avoiding:
        if avoid_timer > 0:
            plane_x += avoid_direction * turning_speed  # Move left (-1) or right (+1)
            avoid_timer -= 1
        else:
            avoiding = False  # Reset avoidance after duration ends
        return

    # Check obstacles and determine avoidance direction
    for obstacle in obstacles:
        if abs(plane_x - obstacle["x"]) < 100 and abs(plane_y - obstacle["y"]) < 100:  # 🚀 EARLY DETECTION AT 100px
            avoiding = True
            avoid_timer = AVOID_DURATION  # Maintain turn for a shorter duration

            if plane_x > obstacle["x"]:
                avoid_direction = 1  # Move right
            else:
                avoid_direction = -1  # Move left
            return

def main():
    global plane_x
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        avoid_collision()  # Auto-avoid obstacles + Move plane upward
        draw()
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sio.disconnect()

if __name__ == "__main__":
    main()
