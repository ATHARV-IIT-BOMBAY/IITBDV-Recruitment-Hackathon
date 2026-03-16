Atharv Pratap Singh Chauhan
25B2253
## 🏎️ Hackathon Submissions & Upgrades

### 1. Path Planning & Controls (PPC)
* **Planner:** Implemented an index-based sorting algorithm to pair left/right cones and compute exact midpoints, generating a continuous, ordered track centerline.
* **Controller:** Implemented a **Pure Pursuit** algorithm with a dynamic lookahead distance (scaled by velocity) and loop wrap-around. Added a **Proportional (P) Controller** for stable throttle/brake management, achieving a clean **23.178s lap** with zero cone hits.

### 2. Simultaneous Localization and Mapping (SLAM)
* **Data Association:** Upgraded from greedy Nearest-Neighbor to the **Hungarian Algorithm (Linear Sum Assignment)** using `scipy.optimize`. This ensures optimal global 1-to-1 matching and rejects severe distance outliers.
* **Localization:** Replaced the basic Euler dead-reckoning with **Runge-Kutta 4th Order (RK4) integration**. This significantly reduces trajectory drift on sharp corners by sampling kinematic derivatives multiple times per time step.
* **Mapping:** Upgraded naive point appending to a **Recursive Mean Filter**. Instead of plotting every noisy measurement, the map dynamically averages new observations into existing landmarks to refine their true position and eliminate "point clouds".
