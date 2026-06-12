import numpy as np
import matplotlib.pyplot as plt
from vehicle_model import VehicleState
from controllers.pure_pursuit import PurePursuitController
from controllers.stanley import StanleyController
from controllers.ai_controller import AIController


def run_simulation(controller_class, path_x, path_y):
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=2.0)
    controller = controller_class(path_x, path_y)

    # Telemetry logging arrays
    history_x, history_y, history_cte = [], [], []

    dt = 0.05
    for _ in range(300):
        # 1. Compute control actions
        accel, steer = controller.compute_control(state)

        # 2. Step physics engine
        state.update(accel, steer, dt)

        # 3. Calculate metrics (Cross-Track Error calculation for statistics)
        dx = [state.x - x for x in path_x]
        dy = [state.y - y for y in path_y]
        cte = np.min(np.hypot(dx, dy))

        # Log telemetry
        history_x.append(state.x)
        history_y.append(state.y)
        history_cte.append(cte)

    return history_x, history_y, history_cte


if __name__ == "__main__":
    # Define reference track
    ref_x = np.linspace(0, 100, 200)
    ref_y = 5.0 * np.sin(ref_x / 10.0)

    # Benchmark all three controllers
    print("Running Pure Pursuit Simulation...")
    pp_x, pp_y, pp_cte = run_simulation(PurePursuitController, ref_x, ref_y)
    
    print("Running Stanley Simulation...")
    st_x, st_y, st_cte = run_simulation(StanleyController, ref_x, ref_y)
    
    # The AI Controller will train itself during initialization, then run
    ai_x, ai_y, ai_cte = run_simulation(AIController, ref_x, ref_y)

    # Print Quantitative Descriptive Statistics
    print("\n--- RESULTS ---")
    print(f"Pure Pursuit Mean CTE: {np.mean(pp_cte):.4f} m | Max CTE: {np.max(pp_cte):.4f} m")
    print(f"Stanley Mean CTE:      {np.mean(st_cte):.4f} m | Max CTE: {np.max(st_cte):.4f} m")
    print(f"AI Model Mean CTE:     {np.mean(ai_cte):.4f} m | Max CTE: {np.max(ai_cte):.4f} m")

    # 2D Matplotlib Plotting Engine
    plt.figure(figsize=(12, 6))
    plt.plot(ref_x, ref_y, 'k--', label="Reference Path", linewidth=2)
    plt.plot(pp_x, pp_y, label="Pure Pursuit", alpha=0.7)
    plt.plot(st_x, st_y, label="Stanley", alpha=0.7)
    plt.plot(ai_x, ai_y, label="AI Neural Network", linewidth=2) # Add AI to the plot
    
    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.title("Autonomous Vehicle Controller Benchmarking Framework")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
