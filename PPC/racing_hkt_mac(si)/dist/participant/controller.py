import numpy as np

def steering(path: list[dict], state: dict):
    if not path:
        return 0.0

    x, y, yaw = state['x'], state['y'], state['yaw']
    vx = max(state['vx'], 1.0) # Prevent zero division

    # 1. Find the closest waypoint index
    distances = [np.hypot(wp['x'] - x, wp['y'] - y) for wp in path]
    closest_idx = np.argmin(distances)

    # 2. Dynamic lookahead distance
    ld = np.clip(0.6 * vx + 1.5, 3.0, 10.0)

    # 3. Find target lookahead point (with wrap-around for the loop)
    target_wp = path[-1]
    for i in range(closest_idx, closest_idx + len(path)):
        wp = path[i % len(path)] # Modulo ensures we loop back to the start
        if np.hypot(wp['x'] - x, wp['y'] - y) >= ld:
            target_wp = wp
            break

    # 4. Pure pursuit geometry
    alpha = np.arctan2(target_wp['y'] - y, target_wp['x'] - x) - yaw
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi # Normalize to [-pi, pi]

    wheelbase = 1.55 # Standard FSAE car wheelbase
    steer = np.arctan2(2.0 * wheelbase * np.sin(alpha), ld)
    
    return np.clip(steer, -0.5, 0.5)

def control(
    path: list[dict],
    state: dict,
    cmd_feedback: dict,
    step: int,
) -> tuple[float, float, float]:
    
    steer = steering(path, state)
    
    # Safe, consistent target speed (approx 14 km/h) to guarantee a finish
    target_speed = 4.0 
    error = target_speed - state['vx']
    
    # Simple P-Controller for throttle/brake
    if error > 0:
        throttle = np.clip(error * 0.5, 0.0, 1.0)
        brake = 0.0
    else:
        throttle = 0.0
        brake = np.clip(-error * 0.5, 0.0, 1.0)

    return float(throttle), float(steer), float(brake)
