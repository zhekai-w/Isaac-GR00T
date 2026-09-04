"""Overlay multiple simple_client_log.npz logs for comparison.

Plots per-joint commanded (raw) arm position over time, stacking every given
log on the same axes in a distinct color so different runs can be compared
directly.

With --ee the six joint panels are replaced by the end-effector (flange) pose
computed from the UR5 DH table.

Usage:
    python scripts/plot_stack_logs.py a.npz b.npz [...] \
        [--second N] [--legend "run A" "run B" ...] [--ee]

    --second N   plot only the first N seconds of each log (default: all)
    --legend ... one label per log, in the same order (default: filenames)
    --ee         plot end-effector flange pose (x, y, z, rpy) instead of joints
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

# These figures are read zoomed-out or pasted into slides, so the matplotlib
# defaults (10 pt) are too small to read.
plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 20,
    "figure.titlesize": 26,
})

JOINT_NAMES = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
EE_NAMES = ["x (m)", "y (m)", "z (m)",
            "roll (rad)", "pitch (rad)", "yaw (rad)"]

# UR5 DH table, in the same joint order as the logs
# ([pan, lift, elbow, wrist1, wrist2, wrist3]), so no reordering is needed.
UR5_A = np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0])
UR5_D = np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823])
UR5_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])


def _dh_matrix(a: float, alpha: float, d: float, theta: np.ndarray) -> np.ndarray:
    """Standard (distal) DH link transform for a whole trajectory at once.

    theta is (N,), the return is (N, 4, 4)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.zeros((theta.shape[0], 4, 4))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 2], T[:, 0, 3] = ct, -st * ca, st * sa, a * ct
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2], T[:, 1, 3] = st, ct * ca, -ct * sa, a * st
    T[:, 2, 1], T[:, 2, 2], T[:, 2, 3] = sa, ca, d
    T[:, 3, 3] = 1.0
    return T


def fk_ee(q: np.ndarray) -> np.ndarray:
    """Forward kinematics: (N, 6) joint positions -> (N, 6) flange pose
    [x, y, z, roll, pitch, yaw] in the base frame.

    The euler columns are unwrapped along time so a trace crossing +-pi draws
    as a continuous line instead of jumping the full range of the axis."""
    T = _dh_matrix(UR5_A[0], UR5_ALPHA[0], UR5_D[0], q[:, 0])
    for j in range(1, 6):
        T = T @ _dh_matrix(UR5_A[j], UR5_ALPHA[j], UR5_D[j], q[:, j])
    rpy = Rotation.from_matrix(T[:, :3, :3]).as_euler("xyz")
    return np.concatenate([T[:, :3, 3], np.unwrap(rpy, axis=0)], axis=1)


def load_log(path: str, seconds: float | None = None, ee: bool = False):
    """Return (t, val, boundary_idx) for one log, shifted to start at t=0 and
    optionally truncated to the first `seconds`. val is the raw commanded joint
    waypoints, or the flange pose derived from them when ee=True. boundary_idx
    marks the first waypoint of each new action chunk."""
    data = np.load(path)
    t = data["t"] - data["t"][0]
    cmd = data["cmd"]      # (N, 6) raw commanded arm waypoints
    chunk = data["chunk"]  # (N,) chunk/cycle id per waypoint
    if seconds is not None:
        mask = t <= seconds
        t, cmd, chunk = t[mask], cmd[mask], chunk[mask]
    boundary_idx = np.flatnonzero(np.diff(chunk) != 0) + 1
    return t, (fk_ee(cmd) if ee else cmd), boundary_idx


def _align_euler_branch(logs: list) -> list:
    """Put every log's euler columns on the same 2*pi branch as the first log.

    These runs sit near roll = +-pi, so two logs at the same physical
    orientation can land on opposite branches and plot 2*pi apart, which makes
    the overlay unreadable. Shift each log by whole turns to sit closest to the
    first log's starting value."""
    ref = logs[0][1][0, 3:]
    out = [logs[0]]
    for t, val, boundary_idx in logs[1:]:
        val = val.copy()
        val[:, 3:] -= 2 * np.pi * np.round((val[0, 3:] - ref) / (2 * np.pi))
        out.append((t, val, boundary_idx))
    return out


def main(paths: list[str], seconds: float | None = None,
         legend: list[str] | None = None, ee: bool = False):
    if legend is None:
        legend = [p.rsplit("/", 1)[-1] for p in paths]
    if len(legend) != len(paths):
        raise SystemExit(
            f"--legend has {len(legend)} labels but {len(paths)} logs given"
        )

    logs = [load_log(p, seconds, ee) for p in paths]
    if ee:
        logs = _align_euler_branch(logs)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    # EE panels mix metres and radians, so the unit lives in the label
    labels = EE_NAMES if ee else [f"{n} (rad)" for n in JOINT_NAMES]

    # Text size is absolute (points) but the canvas is not: a tall figure makes
    # even large labels read as small once the PNG is scaled to fit a screen.
    fig, axes = plt.subplots(6, 1, figsize=(11, 15), constrained_layout=True,
                             sharex=True)
    for j in range(6):
        ax = axes[j]
        for i, (t, val, boundary_idx) in enumerate(logs):
            color = colors[i % 10]
            ax.plot(t, val[:, j], "-o", ms=2, lw=1, color=color, label=legend[i])
            # chunk boundaries for this log, in its own color
            for bi in boundary_idx:
                ax.axvline(t[bi], color=color, ls="--", lw=0.6, alpha=0.4)
        ax.set_ylabel(labels[j])
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")

    what = "end-effector (flange) pose" if ee else "commanded (raw)"
    title = f"{what} — {len(paths)} logs overlaid"
    if seconds is not None:
        title += f", first {seconds:g}s"
    fig.suptitle(title, fontsize=24)

    out = "stacked_logs_ee.png" if ee else "stacked_logs.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="one or more .npz logs to overlay")
    parser.add_argument("--second", type=float, default=None, dest="seconds",
                        help="plot only the first N seconds of each log (default: all)")
    parser.add_argument("--legend", nargs="+", default=None,
                        help="legend label per log, in the same order (default: filenames)")
    parser.add_argument("--ee", action="store_true",
                        help="plot end-effector flange pose (x, y, z, rpy) from "
                             "UR5 DH forward kinematics instead of joint values")
    args = parser.parse_args()
    main(args.paths, args.seconds, args.legend, args.ee)
