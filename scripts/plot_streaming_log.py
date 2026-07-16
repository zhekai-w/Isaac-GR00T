"""Plot the diagnostic log written by ur5_gr00t_streaming_client.py.

Usage:
    python scripts/plot_streaming_log.py [streaming_log.npz]

Produces streaming_log.png with three panels:
  1. Commanded vs measured joint positions over time (per joint)
  2. Commanded per-tick position delta (rad) — staircase/noise in the policy output
  3. Tick interval histogram — control-loop timing jitter
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOINT_NAMES = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]


def main(path: str = "streaming_log.npz"):
    data = np.load(path)
    t = data["t"] - data["t"][0]
    cmd = data["cmd"]    # (N, 6)
    meas = data["meas"]  # (N, 6)
    step = data["step"]

    fig, axes = plt.subplots(8, 1, figsize=(14, 22), constrained_layout=True)

    # Panel 1-6: per-joint commanded vs measured
    for j in range(6):
        ax = axes[j]
        ax.plot(t, cmd[:, j], "-o", ms=2, lw=1, label="commanded", color="tab:blue")
        ax.plot(t, meas[:, j], "-", lw=1, label="measured", color="tab:orange")
        # Mark chunk boundaries (timestep jumps back or inference refills)
        ax.set_ylabel(f"{JOINT_NAMES[j]} (rad)")
        if j == 0:
            ax.legend(loc="upper right")

    # Panel 7: commanded per-tick delta magnitude
    d = np.abs(np.diff(cmd, axis=0))
    axes[6].plot(t[1:], d.max(axis=1), lw=1)
    axes[6].set_ylabel("max |Δcmd| per tick (rad)")
    axes[6].set_xlabel("time (s)")

    # Panel 8: tick interval
    dt_actual = np.diff(t)
    axes[7].plot(t[1:], dt_actual, lw=1)
    axes[7].axhline(np.median(dt_actual), color="k", ls="--", lw=0.8,
                    label=f"median={np.median(dt_actual)*1000:.1f} ms")
    axes[7].set_ylabel("tick interval (s)")
    axes[7].set_xlabel("time (s)")
    axes[7].legend()

    out = path.replace(".npz", ".png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")

    # High-rate /joint_states velocity analysis (if recorded)
    if "js" in data.files and len(data["js"]) > 10:
        js = data["js"]
        t0 = data["t"][0]
        jt = js[:, 0] - t0
        # js is canonical [pan, lift, elbow, w1, w2, w3] — the client callback
        # keys /joint_states by name (logs from before that fix are scrambled).
        jp = js[:, 1:7]
        jv = np.diff(jp, axis=0) / np.diff(jt)[:, None]
        goal_t = data["t"] - t0
        grip_t = data["grip_t"] - t0 if len(data["grip_t"]) else np.array([])

        fig2, axes2 = plt.subplots(6, 1, figsize=(16, 18), constrained_layout=True, sharex=True)
        names_drv = JOINT_NAMES  # js is canonical [pan, lift, elbow, w1, w2, w3]
        for j in range(6):
            ax = axes2[j]
            ax.plot(jt[1:], jv[:, j], lw=0.6, color="tab:blue")
            for gt in goal_t:
                ax.axvline(gt, color="green", lw=0.4, alpha=0.3)
            for gt in grip_t:
                ax.axvline(gt, color="red", lw=0.8, alpha=0.6)
            ax.set_ylabel(f"{names_drv[j]} vel (rad/s)")
        axes2[-1].set_xlabel("time (s)  [green=arm goal sent, red=gripper cmd sent]")
        out2 = path.replace(".npz", "_vel.png")
        fig2.savefig(out2, dpi=120)
        print(f"Saved {out2}")
        print(f"joint_states rate: {1.0 / np.median(np.diff(jt)):.0f} Hz, {len(jt)} samples")
        print(f"gripper commands sent: {len(grip_t)}")
    print(f"ticks: {len(t)}, duration: {t[-1]:.1f}s")
    print(f"tick interval: median {np.median(dt_actual)*1000:.1f} ms, "
          f"p95 {np.percentile(dt_actual, 95)*1000:.1f} ms, "
          f"max {dt_actual.max()*1000:.1f} ms")
    print(f"max per-tick commanded delta: {d.max():.4f} rad "
          f"(joint {JOINT_NAMES[int(np.unravel_index(d.argmax(), d.shape)[1])]})")
    gaps = np.diff(step)
    print(f"timestep continuity: {np.sum(gaps != 1)} non-consecutive jumps of {len(gaps)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "streaming_log.npz")
