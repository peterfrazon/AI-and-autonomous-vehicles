import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from vehicle_model import VehicleState
from controllers.pure_pursuit import PurePursuitController

class SteeringNet(nn.Module):
    def __init__(self):
        super(SteeringNet, self).__init__()
        
        # (CTE, Heading Error, and 5 upcoming local X,Y path points)
        self.network = nn.Sequential(
            nn.Linear(12, 32),  # Wider hidden layer to process the path
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

class AIController:
    def __init__(self, path_x, path_y):
        self.path_x = path_x
        self.path_y = path_y
        self.model = SteeringNet()
        self._train_behavioral_cloning()

    def _train_behavioral_cloning(self):
        print("\n--- Training Predictive AI Controller ---")
        expert_controller = PurePursuitController(self.path_x, self.path_y)
        
        states = []
        actions = []
        dt = 0.05
        
        # Data Augmentation to ensure robustness
        initial_y_offsets = [-1.5, -0.75, 0.0, 0.75, 1.5]
        
        for offset in initial_y_offsets:
            expert_state = VehicleState(x=0.0, y=offset, yaw=0.0, v=2.0)
            step_count = 0
            
            while expert_state.x < self.path_x[-1] and step_count < 2000:
                step_count += 1
                
                # Get the 12-feature state
                state_features = self._get_state_features(expert_state)
                accel, steer = expert_controller.compute_control(expert_state)
                
                states.append(state_features)
                actions.append([steer])
                
                expert_state.update(accel, steer, dt)
                
        X_train = torch.tensor(states, dtype=torch.float32)
        y_train = torch.tensor(actions, dtype=torch.float32)
        
        print(f"   Collected {len(states)} predictive state-action pairs.")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        
        epochs = 500
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f"   Epoch {epoch+1}/{epochs} | Loss: {loss.item():.5f}")
        print("Training Complete!\n")

    def _get_state_features(self, state):
        """Calculates CTE, Heading Error, AND the upcoming path trajectory"""
        dx = [state.x - x for x in self.path_x]
        dy = [state.y - y for y in self.path_y]
        distances = np.hypot(dx, dy)
        closest_idx = np.argmin(distances)
        
        # 1. Base Errors (Reactive)
        next_idx = min(closest_idx + 1, len(self.path_x) - 1)
        path_yaw = np.arctan2(self.path_y[next_idx] - self.path_y[closest_idx],
                              self.path_x[next_idx] - self.path_x[closest_idx])
        
        dx_val = state.x - self.path_x[closest_idx]
        dy_val = state.y - self.path_y[closest_idx]
        cte = dy_val * np.cos(path_yaw) - dx_val * np.sin(path_yaw)
        
        epsi = state.yaw - path_yaw
        epsi = (epsi + np.pi) % (2 * np.pi) - np.pi
        
        # 2. Path Lookahead (Predictive)
        # Sample 5 points ahead on the path (spacing them out by 4 indices each)
        features = [cte, epsi]
        
        for i in range(1, 6):
            target_idx = min(closest_idx + (i * 4), len(self.path_x) - 1)
            global_x = self.path_x[target_idx]
            global_y = self.path_y[target_idx]
            
            # Transform global path coordinates to the vehicle's local perspective
            dx_path = global_x - state.x
            dy_path = global_y - state.y
            
            # Rotation matrix math to align path with car's current heading
            local_x = dx_path * np.cos(-state.yaw) - dy_path * np.sin(-state.yaw)
            local_y = dx_path * np.sin(-state.yaw) + dy_path * np.cos(-state.yaw)
            
            features.extend([local_x, local_y])
            
        return features

    def compute_control(self, state):
        state_features = self._get_state_features(state)
        state_tensor = torch.tensor(state_features, dtype=torch.float32)
        with torch.no_grad():
            steer = self.model(state_tensor).item()
        return 0.0, steer
