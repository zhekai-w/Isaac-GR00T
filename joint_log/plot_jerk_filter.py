"""Measure per-joint jerk of commanded trajectories before vs. after filtering.

Takes one or more simple_client_log.npz logs (each treated as a separate
trajectory), filters the raw commanded arm trajectory with one of the smoothers
in filter_utils, and compares the joint jerk (3rd position derivative) before
and after.

Jerk is computed exactly as in eval_policy.py:
    jerk = np.diff(traj, n=3, axis=0) / dt**3
RMS jerk and the per-timestep jerk magnitude reuse that definition.

Plotting follows plot_stack_logs.py: 6 per-joint subplots with every log
overlaid. Each trajectory gets one base color; the *raw* jerk is drawn in that
color and the *filtered* jerk in a lighter shade of the same color.

Usage:
    python scripts/plot_jerk_filter.py a.npz b.npz [...] \
        [--filter oneeuro|savgol|rts] [--second N] [--legend "A" "B" ...]

    --filter        smoother to apply (default: oneeuro)
    --second N      use only the first N seconds of each log (default: all)
    --legend ...    one label per log, in order (default: filenames)

    One Euro:  --mincutoff 1.0 --beta 0.1
    Savgol:    --window 7 --polyorder 3   (window odd, < trajectory length)
    RTS:       --q 1e-3 --r 1e-4
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

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

sys.path.insert(0, os.path.dirname(__file__))
from filter_utils import OneEuroFilter, savgol_chunk, rts_smoother_chunk

JOINT_NAMES = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]


# --- jerk (matches eval_policy.py) -----------------------------------------
def compute_rms_jerk(traj: np.ndarray, dt: float) -> float:
    """RMS jerk (third derivative of position) over a (T, D) trajectory."""
    jerk = np.diff(traj, n=3, axis=0) / (dt ** 3)
    return float(np.sqrt(np.mean(jerk ** 2)))


def jerk_per_joint(traj: np.ndarray, dt: float) -> np.ndarray:
    """Per-timestep, per-joint jerk. traj: (T, D). Returns (T-3, D)."""
    return np.diff(traj, n=3, axis=0) / (dt ** 3)


# --- filtering --------------------------------------------------------------
def apply_filter(traj: np.ndarray, dt: float, args) -> np.ndarray:
    """Smooth a full (T, D) commanded trajectory with the selected filter."""
    if args.filter == "oneeuro":
        freq = 1.0 / dt
        out = np.empty_like(traj)
        for j in range(traj.shape[1]):
            f = OneEuroFilter(freq, args.mincutoff, args.beta)
            for t in range(traj.shape[0]):
                out[t, j] = f(traj[t, j])
        return out
    if args.filter == "savgol":
        return savgol_chunk(traj, args.window, args.polyorder)
    if args.filter == "rts":
        return rts_smoother_chunk(traj, dt=dt, q=args.q, r=args.r)
    raise ValueError(f"unknown filter {args.filter!r}")


def lighten(color, amount: float = 0.55):
    """Blend an RGB(A) color toward white. amount 0 = same, 1 = white."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# --- loading (matches plot_stack_logs.py) -----------------------------------
def load_log(path: str, seconds: float | None = None):
    """Return (t, cmd) for one log, shifted to start at t=0 and optionally
    truncated to the first `seconds`."""
    data = np.load(path)
    t = data["t"] - data["t"][0]
    cmd = data["cmd"]  # (N, 6) raw commanded arm waypoints
    if seconds is not None:
        mask = t <= seconds
        t, cmd = t[mask], cmd[mask]
    return t, cmd


def main(paths, seconds, legend, args):
    if legend is None:
        legend = [p.rsplit("/", 1)[-1] for p in paths]
    if len(legend) != len(paths):
        raise SystemExit(
            f"--legend has {len(legend)} labels but {len(paths)} logs given"
        )

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(6, 1, figsize=(7, 15), constrained_layout=True,
                             sharex=True)

    print(f"filter = {args.filter}")
    all_stats = []  # (label, rms_raw, rms_filt)
    for i, path in enumerate(paths):
        t, cmd = load_log(path, seconds)
        if len(cmd) < 4:
            print(f"  skip {legend[i]}: only {len(cmd)} samples (<4)")
            continue
        dt = float(np.median(np.diff(t)))
        filt = apply_filter(cmd, dt, args)

        j_raw = jerk_per_joint(cmd, dt)    # (T-3, 6)
        j_filt = jerk_per_joint(filt, dt)  # (T-3, 6)
        tj = t[3:]  # jerk is defined from the 4th sample on

        rms_raw = compute_rms_jerk(cmd, dt)
        rms_filt = compute_rms_jerk(filt, dt)
        reduction = 100 * (rms_raw - rms_filt) / rms_raw if rms_raw > 0 else 0.0
        all_stats.append((legend[i], rms_raw, rms_filt))
        print(f"  {legend[i]}: RMS jerk raw={rms_raw:.2f}  "
              f"filtered={rms_filt:.2f}  ({reduction:.1f}% reduction)")

        base = colors[i % 10]
        light = lighten(base)
        for j in range(6):
            ax = axes[j]
            ax.plot(tj, np.abs(j_raw[:, j]), "-", lw=1, color=base,
                    label=f"{legend[i]} raw" if j == 0 else None)
            ax.plot(tj, np.abs(j_filt[:, j]), "--", lw=1.2, color=light,
                    label=f"{legend[i]} filtered" if j == 0 else None)

    for j in range(6):
        axes[j].set_ylabel(f"{JOINT_NAMES[j]}\n|jerk| (rad/s³)")
        axes[j].set_yscale("log")
        axes[j].grid(True, which="both", alpha=0.3)
    # ncol=1 at this width: constrained_layout cannot shrink a legend, so a
    # wide one steals the space from the axes instead.
    axes[0].legend(loc="upper right", fontsize=12, ncol=1, framealpha=0.9)
    axes[-1].set_xlabel("time (s)")
    # Same pinned x range as plot_interchunk_diff.py: jerk is only defined from
    # the 4th sample on, so autoscaling would start these axes at t~0.2 s and
    # the two figures would not share a scale.
    axes[-1].set_xlim(0.0, seconds if seconds is not None else float(tj[-1]))

    # Wrapped: the one-line form is wider than the 7 in figure at this font size.
    title = f"joint jerk before/after {args.filter} filter\n{len(paths)} logs overlaid"
    if seconds is not None:
        title += f", first {seconds:g}s"
    fig.suptitle(title, fontsize=16)

    out = "jerk_filter.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")

    if all_stats:
        mean_raw = np.mean([s[1] for s in all_stats])
        mean_filt = np.mean([s[2] for s in all_stats])
        red = 100 * (mean_raw - mean_filt) / mean_raw if mean_raw > 0 else 0.0
        print(f"Mean RMS jerk — raw: {mean_raw:.2f}  filtered: {mean_filt:.2f}  "
              f"({red:.1f}% reduction)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="one or more .npz logs to overlay")
    parser.add_argument("--filter", choices=["oneeuro", "savgol", "rts"],
                        default="oneeuro", help="smoother to apply (default: oneeuro)")
    parser.add_argument("--second", type=float, default=None, dest="seconds",
                        help="use only the first N seconds of each log (default: all)")
    parser.add_argument("--legend", nargs="+", default=None,
                        help="legend label per log, in order (default: filenames)")
    # One Euro
    parser.add_argument("--mincutoff", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    # Savitzky-Golay
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--polyorder", type=int, default=3)
    # RTS
    parser.add_argument("--q", type=float, default=1e-3)
    parser.add_argument("--r", type=float, default=1e-4)
    args = parser.parse_args()
    main(args.paths, args.seconds, args.legend, args)
