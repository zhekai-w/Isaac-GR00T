"""Plot the joint-signal log written by ur5_gr00t_simple_client.py (--log) and
calculate the inter action chunk difference.

An action chunk is the (action_horizon, 6) array of arm waypoints returned by
one client.get_action(obs) call in ur5_gr00t_simple_client.py. Chunks execute
back-to-back with no overlap, so the "inter action chunk difference" is the
jump between the last commanded waypoint of chunk N-1 and the first commanded
waypoint of chunk N -- the discontinuity that boundary_blend is meant to
smooth.

Usage:
    python scripts/plot_interchunk_diff.py [simple_client_log.npz]

Produces <path>_interchunk.png with:
  1. Per-joint commanded (raw) position over time, with measured position
     overlaid (derived from the high-rate /joint_states log) and chunk
     boundaries marked
  2. Per-joint inter action chunk difference magnitude at each boundary
"""

import matplotlib

matplotlib.use("Agg")
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

JOINT_NAMES = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]


def main(path: str = "simple_client_log.npz", seconds: float | None = None):
    data = np.load(path)
    t0 = data["t"][0]
    t = data["t"] - t0
    cmd = data["cmd"]      # (N, 6) raw commanded arm waypoints
    chunk = data["chunk"]  # (N,) chunk/cycle id per waypoint

    # Restrict to the first `seconds` of the log if requested
    if seconds is not None:
        mask = t <= seconds
        t, cmd, chunk = t[mask], cmd[mask], chunk[mask]

    boundary_idx = np.flatnonzero(np.diff(chunk) != 0) + 1
    if len(boundary_idx) == 0:
        print("No chunk boundaries found (need >= 2 chunks) -- nothing to compute.")
        return

    # Measured position, derived from the high-rate /joint_states log if present
    meas = None
    if "js" in data.files and len(data["js"]) > 10:
        js = data["js"]
        jt = js[:, 0] - t0
        # js is already canonical [pan, lift, elbow, w1, w2, w3] — the client
        # callback keys /joint_states by name, so no reorder is needed. (Logs
        # captured before that fix are in the scrambled raw order and won't align.)
        js_scaled = js[:, 1:7]
        meas = np.stack([np.interp(t, jt, js_scaled[:, j]) for j in range(6)], axis=1)

    # Inter action chunk difference: last waypoint of chunk N-1 vs first waypoint of chunk N
    boundary_diff = cmd[boundary_idx] - cmd[boundary_idx - 1]  # (num_boundaries, 6)
    mag = np.abs(boundary_diff)

    # sharex + 2.5 in per panel, matching plot_jerk_filter.py (6 panels / 15 in):
    # only the bottom panel carries tick labels.
    fig, axes = plt.subplots(7, 1, figsize=(7, 17.5), constrained_layout=True,
                             sharex=True)

    # Panel 1-6: per-joint commanded (+ measured) position, chunk boundaries marked
    for j in range(6):
        ax = axes[j]
        ax.plot(t, cmd[:, j], "-o", ms=2, lw=1, label="commanded (raw)", color="tab:blue")
        if meas is not None:
            ax.plot(t, meas[:, j], "-", lw=1, label="measured (from js)", color="tab:orange")
        for bi in boundary_idx:
            ax.axvline(t[bi], color="gray", ls="--", lw=0.6, alpha=0.5)
        ax.set_ylabel(f"{JOINT_NAMES[j]} (rad)")
        if j == 0:
            ax.legend(loc="upper right", fontsize=12)

    # Match wrist2's y-axis scale (span) to pan's, but keep it centered on
    # wrist2's own data so the values stay visible.
    pan_lo, pan_hi = axes[JOINT_NAMES.index("pan")].get_ylim()
    span = pan_hi - pan_lo
    w2_ax = axes[JOINT_NAMES.index("wrist2")]
    w2_lo, w2_hi = w2_ax.get_ylim()
    w2_center = 0.5 * (w2_lo + w2_hi)
    w2_ax.set_ylim(w2_center - 0.5 * span, w2_center + 0.5 * span)

    # Panel 7: inter action chunk difference per joint at each boundary
    ax = axes[6]
    boundary_t = t[boundary_idx]
    for j in range(6):
        ax.plot(boundary_t, mag[:, j], "-o", ms=4, lw=1, label=JOINT_NAMES[j])
    # Two lines: at 22 pt this label is taller than its (short) panel and
    # overruns the one above it.
    ax.set_ylabel("inter-chunk\n|Δ| (rad)")
    ax.set_xlabel("time (s)")
    ax.set_ylim(0.0, 0.06)
    # Pin the shared x range to the requested window rather than letting each
    # script autoscale to its own data: jerk plots start at the 4th sample, so
    # autoscaled limits differ by a few percent and the two figures no longer
    # share a scale.
    ax.set_xlim(0.0, seconds if seconds is not None else float(t[-1]))
    ax.legend(loc="upper right", ncol=3, fontsize=12)

    overall_max_j = int(np.unravel_index(mag.argmax(), mag.shape)[1])
    # Wrapped: the one-line form is wider than the 7 in figure at this font size.
    summary = (f"mean max|Δ|={mag.max(axis=1).mean():.4f} rad, "
               f"median={np.median(mag.max(axis=1)):.4f} rad\n"
               f"max={mag.max():.4f} rad (joint {JOINT_NAMES[overall_max_j]})")
    fig.suptitle(summary, fontsize=16)

    out = path.replace(".npz", "_interchunk.png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")

    print(f"inter action chunk difference: {len(boundary_idx)} chunk boundaries")
    for k, bi in enumerate(boundary_idx):
        j = int(np.argmax(mag[k]))
        print(f"  boundary @ t={t[bi]:6.2f}s (chunk {chunk[bi - 1]}->{chunk[bi]}): "
              f"max |Δ|={mag[k, j]:.4f} rad ({JOINT_NAMES[j]})")
    per_joint_mean = ", ".join(f"{JOINT_NAMES[j]}={mag[:, j].mean():.4f}" for j in range(6))
    print(f"  per-joint mean |Δ| (rad): {per_joint_mean}")
    print(f"  mean max|Δ|={mag.max(axis=1).mean():.4f} rad, "
          f"median={np.median(mag.max(axis=1)):.4f} rad, "
          f"max={mag.max():.4f} rad (joint {JOINT_NAMES[overall_max_j]})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="simple_client_log.npz",
                        help="path to the .npz log written by --log")
    parser.add_argument("--second", type=float, default=None, dest="seconds",
                        help="plot only the first N seconds of the log (default: all)")
    args = parser.parse_args()
    main(args.path, args.seconds)
