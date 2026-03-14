"""
sim/visualizer.py  —  PPC Hackathon Visualizer
Real-time Matplotlib animation of the car, cones, path, telemetry and lap info.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from collections import deque


# ── Styling constants ─────────────────────────────────────────────────────────
BG_COLOR        = "#0d0d0d"
GRID_COLOR      = "#1e1e1e"
TEXT_COLOR      = "#e0e0e0"
ACCENT_COLOR    = "#00ffe0"

CONE_LEFT_COLOR  = "#0099ff"   # blue    (left boundary)
CONE_RIGHT_COLOR = "#ffcc00"   # yellow  (right boundary)
PATH_COLOR       = "#00ffe0"   # cyan    (planned path)
TRACE_COLOR      = "#ff8800"   # orange  (driven trace)
CAR_COLOR        = "#ffffff"   # white   (car body)
CAR_DIR_COLOR    = "#00ff88"   # green   (heading arrow)

CONE_SIZE        = 110           # scatter marker size
CAR_LENGTH       = 2.5         # metres (visual only)
CAR_WIDTH        = 1.2
TRACE_LEN        = 300         # how many past positions to show


class Visualizer:
    def __init__(self, cones: list[dict], path: list[dict], dt: float = 0.05):
        """
        Args:
            cones : list of {"x", "y", "side": "left"/"right", "index": int}
            path  : list of {"x", "y"} waypoints
            dt    : simulation timestep in seconds
        """
        self.cones = cones
        self.path  = path
        self.dt    = dt

        # History buffers
        self.trace_x   = deque(maxlen=TRACE_LEN)
        self.trace_y   = deque(maxlen=TRACE_LEN)
        self.speed_hist = deque(maxlen=200)
        self.steer_hist = deque(maxlen=200)
        self.throttle_hist = deque(maxlen=200)
        self.brake_hist    = deque(maxlen=200)
        self.time_hist     = deque(maxlen=200)

        self.step      = 0
        self.lap_time  = 0.0
        self.last_cmd  = (0.0, 0.0, 0.0)   # throttle, steer, brake
        self.status    = "RUNNING"

        self._build_figure()

    # ── Figure layout ─────────────────────────────────────────────────────────
    def _build_figure(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)
        self.fig.canvas.manager.set_window_title("🏎  PPC Hackathon — Live Simulation")

        gs = gridspec.GridSpec(
            3, 3,
            figure=self.fig,
            left=0.05, right=0.97,
            top=0.93,  bottom=0.07,
            wspace=0.35, hspace=0.5
        )

        # Main track view (left 2/3)
        self.ax_track = self.fig.add_subplot(gs[:, :2])
        self._style_ax(self.ax_track, title="TRACK VIEW")

        # Telemetry panels (right column)
        self.ax_speed    = self.fig.add_subplot(gs[0, 2])
        self.ax_steer    = self.fig.add_subplot(gs[1, 2])
        self.ax_cmd      = self.fig.add_subplot(gs[2, 2])

        self._style_ax(self.ax_speed, title="SPEED  (m/s)")
        self._style_ax(self.ax_steer, title="STEER ANGLE  (rad)")
        self._style_ax(self.ax_cmd,   title="THROTTLE / BRAKE")

        # Title bar
        self.fig.text(0.5, 0.97, "PPC HACKATHON  ·  LIVE SIM",
                      ha="center", va="top",
                      fontsize=13, color=ACCENT_COLOR,
                      fontfamily="monospace", fontweight="bold")

        self.lap_text = self.fig.text(
            0.5, 0.005, "LAP TIME: 0.000 s   |   STATUS: RUNNING",
            ha="center", va="bottom",
            fontsize=10, color=TEXT_COLOR, fontfamily="monospace"
        )

        self._draw_static()
        self._init_dynamic()

    def _style_ax(self, ax, title=""):
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--")
        if title:
            ax.set_title(title, color=ACCENT_COLOR,
                         fontsize=8, fontfamily="monospace", pad=4)

    # ── Static elements (drawn once) ──────────────────────────────────────────
    def _draw_static(self):
        ax = self.ax_track

        left_cones   = [c for c in self.cones if c["side"] == "left"]
        right_cones  = [c for c in self.cones if c["side"] == "right"]
        orange_cones = [c for c in self.cones if c["side"] == "orange"]

        # ── Boundaries ──
        for group in [left_cones, right_cones]:
            if not group:
                continue
            xs = [c["x"] for c in group]
            ys = [c["y"] for c in group]
            xs.append(group[0]["x"])  # close the loop
            ys.append(group[0]["y"])
            print(group)
            color = group[0]["color"]           # ← read from cone
            ax.scatter(xs, ys, color=color, s=CONE_SIZE+0,
                    marker="o", zorder=4)
            ax.plot(xs, ys, color=color, alpha=0.25,
                    linewidth=0.8, zorder=3)

        # ── Planned path (waypoints) ──
        if self.path:
            px = [wp["x"] for wp in self.path]
            py = [wp["y"] for wp in self.path]

            # draw as a dashed/cyan line behind other elements
            ax.scatter(px, py, color=PATH_COLOR, s=20,
                    marker=".", zorder=4)

        # ── Orange start/finish ──
        if orange_cones:
            xs = [c["x"] for c in orange_cones]
            ys = [c["y"] for c in orange_cones]
            color = orange_cones[0]["color"]    # ← read from cone
            ax.scatter(xs, ys, color=color, s=CONE_SIZE,
                    marker="^", zorder=4,
                    edgecolors="#ffffff", linewidths=1.0,
                    label="Start/Finish")
            if len(xs) >= 2:
                ax.plot([xs[0], xs[1]], [ys[0], ys[1]],
                        color=color, linewidth=3.0, zorder=9)
            if len(xs) >= 4:
                ax.plot([xs[2], xs[3]], [ys[2], ys[3]],
                        color=color, linewidth=3.0, zorder=9)
                
                
    # ── Dynamic elements (updated each frame) ─────────────────────────────────
    def _init_dynamic(self):
        ax = self.ax_track

        # Driven trace
        self.trace_line, = ax.plot([], [], color=TRACE_COLOR,
                                   linewidth=1.5, alpha=0.7, zorder=5)


        # Car rectangle
        self.car_patch = patches.FancyBboxPatch(
            (0, 0), CAR_LENGTH, CAR_WIDTH,
            boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor=ACCENT_COLOR,
            facecolor=CAR_COLOR, alpha=0.9, zorder=7
        )
        ax.add_patch(self.car_patch)

        # Heading arrow
        self.heading_arrow = FancyArrowPatch(
            (0, 0), (0, 0),
            arrowstyle="-|>",
            color=CAR_DIR_COLOR,
            linewidth=2, mutation_scale=12, zorder=8
        )
        ax.add_patch(self.heading_arrow)

        # Car dot (always visible regardless of zoom)
        self.car_dot, = ax.plot([], [], "o", color=ACCENT_COLOR,
                                markersize=6, zorder=9)

        # Speed / steer / cmd telemetry lines
        self.speed_line,    = self.ax_speed.plot([], [], color="#00ffe0", lw=1.2)
        self.steer_line,    = self.ax_steer.plot([], [], color="#ff8800", lw=1.2)
        self.throttle_line, = self.ax_cmd.plot([], [], color="#44ff88",
                                               lw=1.2, label="Throttle")
        self.brake_line,    = self.ax_cmd.plot([], [], color="#ff4444",
                                               lw=1.2, label="Brake")
        self.ax_cmd.legend(loc="upper right", fontsize=6,
                           facecolor="#1a1a1a", edgecolor=GRID_COLOR,
                           labelcolor=TEXT_COLOR)

        # HUD text (position overlay on track)
        self.hud_text = ax.text(
            0.02, 0.97, "",
            transform=ax.transAxes,
            fontsize=8, color=TEXT_COLOR,
            fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#1a1a1a", edgecolor=GRID_COLOR, alpha=0.85)
        )

    # ── Public update call (called by run.py every step) ──────────────────────
    def update(self, true_state: np.ndarray, step: int,
               cmd: tuple = (0.0, 0.0, 0.0)):
        """
        Args:
            true_state : [x, y, yaw, vx, vy, yaw_rate]
            step       : current simulation step index
            cmd        : (throttle, steer, brake) applied this step
        """
        x, y, yaw, vx, vy, _ = true_state
        throttle, steer, brake = cmd

        self.step     = step
        self.lap_time = step * self.dt
        self.last_cmd = cmd

        speed = np.hypot(vx, vy)

        # ── History ──
        self.trace_x.append(x)
        self.trace_y.append(y)
        self.speed_hist.append(speed)
        self.steer_hist.append(steer)
        self.throttle_hist.append(throttle)
        self.brake_hist.append(brake)
        self.time_hist.append(self.lap_time)

        t_arr = np.array(self.time_hist)

        # ── Trace ──
        self.trace_line.set_data(list(self.trace_x), list(self.trace_y))

        # # ── Path (in case it changed)
        # if hasattr(self, "path_line") and self.path:
        #     px = [wp["x"] for wp in self.path]
        #     py = [wp["y"] for wp in self.path]
        #     self.path_line.set_data(px, py)

        # ── Car rectangle (rotated) ──
        self._update_car_patch(x, y, yaw)

        # ── Heading arrow ──
        arrow_len = CAR_LENGTH * 1.2
        self.heading_arrow.set_positions(
            (x, y),
            (x + arrow_len * np.cos(yaw), y + arrow_len * np.sin(yaw))
        )

        # ── Car dot ──
        self.car_dot.set_data([x], [y])

        # ── Telemetry plots ──
        s_arr  = np.array(self.speed_hist)
        st_arr = np.array(self.steer_hist)
        th_arr = np.array(self.throttle_hist)
        br_arr = np.array(self.brake_hist)

        self.speed_line.set_data(t_arr, s_arr)
        self.steer_line.set_data(t_arr, st_arr)
        self.throttle_line.set_data(t_arr, th_arr)
        self.brake_line.set_data(t_arr, br_arr)

        for ax, arr in [
            (self.ax_speed, s_arr),
            (self.ax_steer, st_arr),
            (self.ax_cmd,   np.concatenate([th_arr, br_arr]))
        ]:
            if len(t_arr) > 1:
                ax.set_xlim(t_arr[0], max(t_arr[-1], t_arr[0] + 1))
            ylo, yhi = arr.min(), arr.max()
            margin = max(abs(yhi - ylo) * 0.15, 0.05)
            ax.set_ylim(ylo - margin, yhi + margin)

        # ── HUD ──
        self.hud_text.set_text(
            f"  t   : {self.lap_time:6.2f} s\n"
            f"  pos : ({x:6.1f}, {y:6.1f}) m\n"
            f"  yaw : {np.degrees(yaw):6.1f} °\n"
            f"  spd : {speed:6.2f} m/s  ({speed*3.6:.1f} km/h)\n"
            f"  thr : {throttle:.2f}  |  brk : {brake:.2f}\n"
            f"  str : {np.degrees(steer):+.2f} °"
        )

        # ── Status bar ──
        self.lap_text.set_text(
            f"LAP TIME: {self.lap_time:.3f} s   |   "
            f"STEP: {step:04d}   |   STATUS: {self.status}"
        )

        plt.pause(0.001)

    def _update_car_patch(self, x, y, yaw):
        self.car_patch.remove()

        self.car_patch = patches.FancyBboxPatch(
            (-CAR_LENGTH / 2, -CAR_WIDTH / 2),  # centered at origin
            CAR_LENGTH, CAR_WIDTH,
            boxstyle="round,pad=0.1",
            linewidth=1.5,
            edgecolor=ACCENT_COLOR,
            facecolor=CAR_COLOR,
            alpha=0.9, zorder=7,
            transform=plt.matplotlib.transforms.Affine2D()
                        .rotate(yaw)               # 1. rotate around origin
                        .translate(x, y)           # 2. move to car position
                    + self.ax_track.transData      # 3. into axes/display coords
        )
        self.ax_track.add_patch(self.car_patch)

    # ── Terminal state display ─────────────────────────────────────────────────
    def show_final(self, status: str = "COMPLETE", lap_time: float = None, lap_summary: str = None):
        """
        Call after simulation ends.
        status: "COMPLETE" | "DNF" | "TIMEOUT"
        """
        self.status = status
        color = {"COMPLETE": "#00ff88", "DNF": "#ff4444",
                 "TIMEOUT": "#ffaa00"}.get(status, TEXT_COLOR)

        t = lap_time if lap_time is not None else self.lap_time
        self.lap_text.set_text(
            lap_summary if lap_summary else
            f"LAP TIME: {t:.3f} s   |   STATUS: {status}"
        )
        self.lap_text.set_color(color)

        # Big centred overlay
        self.ax_track.text(
            0.5, 0.5,
            f"{'✅ LAP COMPLETE' if status == 'COMPLETE' else '❌ ' + status}\n"
            f"{t:.3f} s",
            transform=self.ax_track.transAxes,
            fontsize=22, color=color,
            fontfamily="monospace", fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#000000cc", edgecolor=color, linewidth=2)
        )

        plt.draw()
        plt.pause(0.001)
        plt.show()

    # ── Optional: standalone test ─────────────────────────────────────────────
    @staticmethod
    def demo():
        """Quick smoke-test with a synthetic oval track."""
        import math

        # Generate oval cones
        cones = []
        N = 30
        for i in range(N):
            a = 2 * math.pi * i / N
            cones.append({"x": 20 * math.cos(a), "y": 10 * math.sin(a),
                          "side": "left",  "index": i, "color": CONE_LEFT_COLOR})
            cones.append({"x": 25 * math.cos(a), "y": 14 * math.sin(a),
                          "side": "right", "index": i, "color": CONE_RIGHT_COLOR})

        path = [{"x": 22.5 * math.cos(2 * math.pi * i / 100),
                 "y": 12.0 * math.sin(2 * math.pi * i / 100)}
                for i in range(100)]

        viz = Visualizer(cones, path, dt=0.05)

        # Simulate car driving the oval
        state = np.array([22.5, 0.0, math.pi / 2, 8.0, 0.0, 0.0])
        for step in range(400):
            angle = 2 * math.pi * step / 400
            state[0] = 22.5 * math.cos(angle)
            state[1] = 12.0 * math.sin(angle)
            state[2] = angle + math.pi / 2
            state[3] = 8.0
            cmd = (0.4, math.sin(angle) * 0.15, 0.0)
            viz.update(state, step, cmd)

        viz.show_final("COMPLETE", lap_time=step * 0.05)


# ── Run demo if executed directly ─────────────────────────────────────────────
if __name__ == "__main__":
    Visualizer.demo()