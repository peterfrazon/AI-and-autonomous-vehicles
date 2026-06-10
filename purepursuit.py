import numpy as np
import math
import matplotlib.pyplot as plt

# --- Controller Parameters ---
LOOKAHEAD_DISTANCE = 2.0  # [m] Fixed look-ahead distance
WHEELBASE = 2.5           # [m] Distance between rear and front axle
MAX_STEER = np.radians(30)# [rad] Maximum steering angle

class Vehicle:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, throttle, delta, dt):
        """
        Updates the vehicle's state using a simple kinematic bicycle model.
        Assumes the control point is the rear axle.
        """
        # Clamp steering angle to physical limits
        delta = np.clip(delta, -MAX_STEER, MAX_STEER)

        # Kinematic update equations
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WHEELBASE * math.tan(delta) * dt
        self.v += throttle * dt

def calc_target_index(vehicle, cx, cy):
    """
    Finds the target point on the path at the look-ahead distance.
    """
    # 1. Find the closest point on the path to the vehicle
    dx = [vehicle.x - icx for icx in cx]
    dy = [vehicle.y - icy for icy in cy]
    distances = np.hypot(dx, dy)
    closest_index = np.argmin(distances)

    # 2. Search forward from the closest point to find the look-ahead point
    target_index = closest_index
    while target_index < len(cx):
        # Calculate distance from the rear axle to the current path point
        dist = math.hypot(cx[target_index] - vehicle.x, cy[target_index] - vehicle.y)
        
        # If the distance is greater than the look-ahead distance, we found our target
        if dist >= LOOKAHEAD_DISTANCE:
            break
        target_index += 1

    # If we run out of path points, target the very last point
    if target_index >= len(cx):
        target_index = len(cx) - 1

    return target_index

def pure_pursuit_control(vehicle, tx, ty):
    """
    Calculates the steering angle using the Pure Pursuit algorithm.
    """
    # Calculate the angle to the target point
    alpha = math.atan2(ty - vehicle.y, tx - vehicle.x) - vehicle.yaw
    
    # Pure Pursuit steering angle equation
    delta = math.atan2(2.0 * WHEELBASE * math.sin(alpha), LOOKAHEAD_DISTANCE)
    
    return delta

def main():
    # --- 1. Generate a Reference Path (Sine Wave) ---
    cx = np.arange(0, 50, 0.5)
    cy = [math.sin(ix / 5.0) * (ix / 5.0) for ix in cx]

    # --- 2. Initialize Vehicle and Simulation Parameters ---
    vehicle = Vehicle(x=0.0, y=-3.0, yaw=0.0, v=2.0) # Start slightly off-path
    dt = 0.1  # [s] time step
    time = 0.0
    max_time = 30.0

    # History arrays for plotting
    x_history = []
    y_history = []

    # --- 3. Simulation Loop ---
    while time < max_time:
        # Find the target waypoint
        target_idx = calc_target_index(vehicle, cx, cy)
        tx, ty = cx[target_idx], cy[target_idx]

        # Calculate control input (steering angle)
        delta = pure_pursuit_control(vehicle, tx, ty)

        # Apply control to the vehicle (constant throttle for simplicity)
        vehicle.update(throttle=0.0, delta=delta, dt=dt)

        # Record history
        x_history.append(vehicle.x)
        y_history.append(vehicle.y)
        time += dt

        # Stop if we reached the end of the path
        if target_idx >= len(cx) - 1:
            print("Goal reached!")
            break

        # --- Real-time Plotting ---
        plt.cla()
        plt.plot(cx, cy, "-r", label="Reference Path")
        plt.plot(x_history, y_history, "-b", label="Vehicle Trajectory")
        plt.plot(tx, ty, "go", label="Look-ahead Target")
        
        # Draw vehicle as a simple box/line
        plt.plot(vehicle.x, vehicle.y, "ks", markersize=8)
        
        plt.title(f"Pure Pursuit Tracking | Speed: {vehicle.v} m/s")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.001)

    plt.show()

if __name__ == '__main__':
    main()
