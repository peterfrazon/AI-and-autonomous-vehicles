import numpy as np
import matplotlib.pyplot as plt
from vehicle_model import VehicleState
from controllers.ai_controller import AIController

def run_simulation(controller_class, path_x, path_y):
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=2.0)
    controller = controller_class(path_x, path_y)

    history_x, history_y, history_cte = [], [], []

    dt = 0.05
    step_count = 0
    # Dynamic while loop runs until the car finishes the track
    while state.x < path_x[-1] and step_count < 2000:
        step_count += 1
        
        accel, steer = controller.compute_control(state)
        state.update(accel, steer, dt)

        dx = [state.x - x for x in path_x]
        dy = [state.y - y for y in path_y]
        cte = np.min(np.hypot(dx, dy))

        history_x.append(state.x)
        history_y.append(state.y)
        history_cte.append(cte)

    return history_x, history_y, history_cte

if __name__ == "__main__":
    # Define reference track
    ref_x = np.linspace(0, 100, 200)
    ref_y = 5.0 * np.sin(ref_x / 10.0)

    print("Initializing and Training AI Neural Network...")
    print("(The AI is currently learning from the background expert...)")
    
    # Run the AI simulation
    ai_x, ai_y, ai_cte = run_simulation(AIController, ref_x, ref_y)

    # Print Quantitative Descriptive Statistics
    print("\n--- PERFORMANCE RESULTS ---")
    print(f"AI Model Mean CTE: {np.mean(ai_cte):.4f} m | Max CTE: {np.max(ai_cte):.4f} m")

    # 2D Matplotlib Plotting Engine
    plt.figure(figsize=(12, 6))
    plt.plot(ref_x, ref_y, 'k--', label="Reference Path", linewidth=2)
    plt.plot(ai_x, ai_y, label="AI Neural Network", linewidth=2.5, color='green')
    
    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.title("AI Autonomous Vehicle Controller Evaluation")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
