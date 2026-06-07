import numpy as np


class VehicleState:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, wheelbase=1.530):
        self.x = x
        self.y = y
        self.yaw = yaw  # Orientation in radians
        self.v = v  # Velocity in m/s
        self.L = wheelbase  # Wheelbase (e.g., typical Formula Student scale)

        # Physical constraints
        self.max_steer = np.radians(30.0)  # 30 degrees max lock

    def update(self, acceleration, steering_angle, dt=0.05):
        """
        Updates the vehicle state using the Kinematic Bicycle Model.
        """
        # Enforce steering constraints
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        # Kinematic equations
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += (self.v / self.L) * np.tan(steering_angle) * dt
        self.v += acceleration * dt