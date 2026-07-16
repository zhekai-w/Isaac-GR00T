"""Overlay multiple simple_client_log.npz logs for comparison.

Plots per-joint commanded (raw) arm position over time, stacking every given
log on the same axes in a distinct color so different runs can be compared
directly.

Usage:
    python scripts/plot_stack_logs.py a.npz b.npz [...] \
        [--second N] [--legend "run A" "run B" ...]

    --second N   plot only the first N seconds of each log (default: all)
    --legend ... one label per log, in the same order (default: filenames)
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOINT_NAMES = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]


def load_log(path: str, seconds: float | None = None):
    """Return (t, cmd, boundary_idx) for one log, shifted to start at t=0 and
    optionally truncated to the first `seconds`. boundary_idx marks the first
    waypoint of each new action chunk."""
    data = np.load(path)
    t = data["t"] - data["t"][0]
    cmd = data["cmd"]      # (N, 6) raw commanded arm waypoints
    chunk = data["chunk"]  # (N,) chunk/cycle id per waypoint
    if seconds is not None:
        mask = t <= seconds
        t, cmd, chunk = t[mask], cmd[mask], chunk[mask]
    boundary_idx = np.flatnonzero(np.diff(chunk) != 0) + 1
    return t, cmd, boundary_idx


def main(paths: list[str], seconds: float | None = None,
         legend: list[str] | None = None):
    if legend is None:
        legend = [p.rsplit("/", 1)[-1] for p in paths]
    if len(legend) != len(paths):
        raise SystemExit(
            f"--legend has {len(legend)} labels but {len(paths)} logs given"
        )

    logs = [load_log(p, seconds) for p in paths]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(6, 1, figsize=(14, 20), constrained_layout=True,
                             sharex=True)
    for j in range(6):
        ax = axes[j]
        for i, (t, cmd, boundary_idx) in enumerate(logs):
            color = colors[i % 10]
            ax.plot(t, cmd[:, j], "-o", ms=2, lw=1, color=color, label=legend[i])
            # chunk boundaries for this log, in its own color
            for bi in boundary_idx:
                ax.axvline(t[bi], color=color, ls="--", lw=0.6, alpha=0.4)
        ax.set_ylabel(f"{JOINT_NAMES[j]} (rad)")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")

    title = f"commanded (raw) — {len(paths)} logs overlaid"
    if seconds is not None:
        title += f", first {seconds:g}s"
    fig.suptitle(title, fontsize=12)

    out = "stacked_logs.png"
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
    args = parser.parse_args()
    main(args.paths, args.seconds, args.legend)
