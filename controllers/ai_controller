import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from vehicle_model import VehicleState
from controllers.pure_pursuit import PurePursuitController # Using this as the expert

# 1. Define the Neural Network Architecture
class SteeringNet(nn.Module):
    def __init__(self):
        super(SteeringNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),   # Input: CTE, Heading Error
            nn.ReLU(),
            nn.Linear(16, 16),  # Hidden Layer
            nn.ReLU(),
            nn.Linear(16, 1)    # Output: Steering Angle
        )

    def forward(self, x):
        return self.network(x)

# 2. Define the Controller wrapper expected by main.py
class AIController:
    def __init__(self, path_x, path_y):
        self.path_x = path_x
        self.path_y = path_y
        self.model = SteeringNet()
        
        # Train the model immediately when initialized
        self._train_behavioral_cloning()

    def _train_behavioral_cloning(self):
        print("\n--- Training AI Controller ---")
        print("1. Generating Expert Data (Pure Pursuit)...")
        expert_controller = PurePursuitController(self.path_x, self.path_y)
        expert_state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=2.0)
        
        states = []
        actions = []
        
        dt = 0.05
        # Run a background simulation to collect data
        for _ in range(300):
            # Calculate CTE and Heading Error (The AI's inputs)
            cte, epsi = self._get_state_features(expert_state)
            
            # Get Expert's action
            accel, steer = expert_controller.compute_control(expert_state)
            
            states.append([cte, epsi])
            actions.append([steer])
            
            # Move the expert car forward
            expert_state.update(accel, steer, dt)
            
        X_train = torch.tensor(states, dtype=torch.float32)
        y_train = torch.tensor(actions, dtype=torch.float32)
        
        print("2. Training Neural Network...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        epochs = 250
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 50 == 0:
                print(f"   Epoch {epoch+1}/{epochs} | Loss: {loss.item():.5f}")
        print("Training Complete!\n")

    def _get_state_features(self, state):
        """Helper to calculate Cross-Track Error and Heading Error"""
        dx = [state.x - x for x in self.path_x]
        dy = [state.y - y for y in self.path_y]
        distances = np.hypot(dx, dy)
        closest_idx = np.argmin(distances)
        
        # Calculate Path Heading (Yaw)
        next_idx = min(closest_idx + 1, len(self.path_x) - 1)
        path_yaw = np.arctan2(self.path_y[next_idx] - self.path_y[closest_idx], 
                              self.path_x[next_idx] - self.path_x[closest_idx])
        
        # Calculate CTE
        dx_val = state.x - self.path_x[closest_idx]
        dy_val = state.y - self.path_y[closest_idx]
        cte = dy_val * np.cos(path_yaw) - dx_val * np.sin(path_yaw)
        
        # Calculate Heading Error (epsi)
        epsi = state.yaw - path_yaw
        epsi = (epsi + np.pi) % (2 * np.pi) - np.pi
        
        return cte, epsi

    def compute_control(self, state):
        """Standard method called by run_simulation in main.py"""
        # 1. See the current state
        cte, epsi = self._get_state_features(state)
        
        # 2. Ask the neural network for the steering angle
        state_tensor = torch.tensor([cte, epsi], dtype=torch.float32)
        with torch.no_grad():
            steer = self.model(state_tensor).item()
            
        # 3. Return (acceleration, steering)
        return 0.0, steer
