import numpy as np

def plan(cones: list[dict]) -> list[dict]:
    # Sort the cones purely by the index provided by the simulation
    blue = sorted([c for c in cones if c["side"] == "left"], key=lambda c: c["index"])
    yellow = sorted([c for c in cones if c["side"] == "right"], key=lambda c: c["index"])

    path = []
    # Zip pairs them up beautifully
    for b, y in zip(blue, yellow):
        mid_x = (b["x"] + y["x"]) / 2.0
        mid_y = (b["y"] + y["y"]) / 2.0
        path.append({"x": mid_x, "y": mid_y})

    return path
