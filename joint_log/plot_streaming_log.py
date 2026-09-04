"""Plot the diagnostic log written by ur5_gr00t_streaming_client.py.

Usage:
    python scripts/plot_streaming_log.py [streaming_log.npz]

Produces streaming_log.png with three panels:
  1. Commanded vs measured joint positions over time (per joint)
  2. Commanded per-tick position delta (rad) — staircase/noise in the policy output
  3. Tick interval histogram — control-loop timing jitter

If the log contains `stream_cmd` (the dense setpoint stream published to
forward_position_controller at stream_hz), also produces streaming_log_stream.png
and prints the smoothness metrics used to A/B --segment-interp / --stream-filter.
"""

import sys

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


def smoothness(pos: np.ndarray, dt: float) -> dict:
    """Derivative-based smoothness metrics for a uniformly sampled trajectory.

    RMS jerk alone does not separate a lerp from a spline: the lerp is jerk-free
    inside each segment but fires an impulse at every waypoint, while a cubic has
    a modest jerk everywhere, and pooling squares them washes the difference out.
    max|jerk| and the per-sample velocity step are what actually distinguish them.
    """
    vel = np.diff(pos, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return {
        "rms_jerk": float(np.sqrt(np.mean(jerk ** 2))),
        "max_jerk": float(np.abs(jerk).max()),
        "rms_acc": float(np.sqrt(np.mean(acc ** 2))),
        "max_acc": float(np.abs(acc).max()),
        "max_dvel": float(np.abs(np.diff(vel, axis=0)).max()),
        "max_dpos": float(np.abs(np.diff(pos, axis=0)).max()),
    }


def _print_smoothness(label: str, m: dict) -> None:
    print(f"  {label:24s} rmsJ={m['rms_jerk']:10.1f}  maxJ={m['max_jerk']:11.1f}  "
          f"rmsA={m['rms_acc']:8.2f}  max|Δv|={m['max_dvel']:.4f}  "
          f"max|Δp|={m['max_dpos']:.5f} rad")


def main(path: str = "streaming_log.npz"):
    data = np.load(path)
    t = data["t"] - data["t"][0]
    cmd = data["cmd"]    # (N, 6)
    step = data["step"]

    # Measured trace. Streaming logs carry a per-tick `meas` column; simple_client
    # logs do not, but both carry the full-rate `js` array, so fall back to that.
    # js is plotted on its OWN timebase rather than resampled onto the waypoint
    # schedule: simple_client's `t` has a gap at every chunk boundary (it blocks on
    # the goal result plus a 1 s sleep), and resampling would either break the line
    # there or draw a straight segment across each dwell.
    has_js = "js" in data.files and len(data["js"]) > 10
    if "meas" in data.files:
        meas_t, meas_p = t, data["meas"]
    elif has_js:
        meas_t, meas_p = data["js"][:, 0] - data["t"][0], data["js"][:, 1:7]
    else:
        meas_t = meas_p = None

    fig, axes = plt.subplots(8, 1, figsize=(11, 16), constrained_layout=True)

    # Panel 1-6: per-joint commanded vs measured
    for j in range(6):
        ax = axes[j]
        ax.plot(t, cmd[:, j], "-o", ms=2, lw=1, label="commanded", color="tab:blue")
        if meas_t is not None:
            ax.plot(meas_t, meas_p[:, j], "-", lw=1, label="measured", color="tab:orange")
        # Mark chunk boundaries (timestep jumps back or inference refills)
        ax.set_ylabel(f"{JOINT_NAMES[j]} (rad)")
        if j == 0:
            ax.legend(loc="upper right")

    # Panel 7: commanded per-tick delta magnitude
    d = np.abs(np.diff(cmd, axis=0))
    axes[6].plot(t[1:], d.max(axis=1), lw=1)
    axes[6].set_ylabel("max |Δcmd|\nper tick (rad)")
    axes[6].set_xlabel("time (s)")

    # Panel 8: tick interval
    dt_actual = np.diff(t)
    axes[7].plot(t[1:], dt_actual, lw=1)
    axes[7].axhline(np.median(dt_actual), color="k", ls="--", lw=0.8,
                    label=f"median={np.median(dt_actual)*1000:.1f} ms")
    axes[7].set_ylabel("tick\ninterval (s)")
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
        grip_t = (data["grip_t"] - t0 if "grip_t" in data.files and len(data["grip_t"])
                  else np.array([]))

        # Chunk-merge instants: if roughness is caused by inter-chunk blending
        # rather than by the policy or the controller, it lines up with these.
        merge_t = (data["merge"][:, 0] - t0 if "merge" in data.files
                   and len(data["merge"]) else np.array([]))

        fig2, axes2 = plt.subplots(6, 1, figsize=(12, 13), constrained_layout=True, sharex=True)
        names_drv = JOINT_NAMES  # js is canonical [pan, lift, elbow, w1, w2, w3]
        for j in range(6):
            ax = axes2[j]
            ax.plot(jt[1:], jv[:, j], lw=0.6, color="tab:blue")
            for gt in goal_t:
                ax.axvline(gt, color="green", lw=0.4, alpha=0.3)
            for gt in grip_t:
                ax.axvline(gt, color="red", lw=0.8, alpha=0.6)
            for mt in merge_t:
                ax.axvline(mt, color="purple", lw=1.2, alpha=0.7, ls="--")
            ax.set_ylabel(f"{names_drv[j]} vel (rad/s)")
        axes2[-1].set_xlabel("time (s)  [green=arm goal, red=gripper cmd, "
                             "purple dashed=chunk merge]")
        out2 = path.replace(".npz", "_vel.png")
        fig2.savefig(out2, dpi=120)
        print(f"Saved {out2}")
        print(f"joint_states rate: {1.0 / np.median(np.diff(jt)):.0f} Hz, {len(jt)} samples")
        print(f"gripper commands sent: {len(grip_t)}")
    print(f"ticks: {len(t)}, duration: {t[-1]:.1f}s")
    if "meas" not in data.files:
        print("note: simple_client log — `t` is the NOMINAL schedule "
              "(t_cycle_start + i*dt), not measured dispatch times, so the tick "
              "interval below says nothing about timing jitter")
    print(f"tick interval: median {np.median(dt_actual)*1000:.1f} ms, "
          f"p95 {np.percentile(dt_actual, 95)*1000:.1f} ms, "
          f"max {dt_actual.max()*1000:.1f} ms")
    print(f"max per-tick commanded delta: {d.max():.4f} rad "
          f"(joint {JOINT_NAMES[int(np.unravel_index(d.argmax(), d.shape)[1])]})")
    gaps = np.diff(step)
    print(f"timestep continuity: {np.sum(gaps != 1)} non-consecutive jumps of {len(gaps)}")

    # ------------------------------------------------------------------
    # Dense published setpoint stream — the only place the interpolator and
    # the output filter are visible. `cmd` above is the waypoint stream and
    # is unchanged by either of them.
    # ------------------------------------------------------------------
    cfg = " ".join(
        f"{k}={data[k]}" for k in
        ("control_interface", "segment_interp", "stream_filter", "stream_filter_cutoff",
         "jtc_end_velocity", "aggregate_fn_name", "dt", "stream_hz")
        if k in data.files
    )
    if cfg:
        print(f"\nconfig: {cfg}")

    # Separating "rough trajectory" from "rough control": `cmd` is the waypoint
    # stream the policy asked for (post chunk-filter, post blending), `js` is what
    # the arm actually did. If cmd is already rough, the problem is upstream of
    # the controller; if cmd is smooth and js is not, it is tracking/timing.
    _print_smoothness("waypoints (cmd)", smoothness(cmd, float(np.median(dt_actual))))
    if "js" in data.files and len(data["js"]) > 10:
        _print_smoothness("measured (js)",
                          smoothness(data["js"][:, 1:7],
                                     float(np.median(np.diff(data["js"][:, 0])))))

    # Chunk merges: rate, overlap size, and how much of each chunk is discarded
    # because inference outran the queue.
    if "merge" in data.files and len(data["merge"]) > 1:
        mg = data["merge"]
        gap = np.diff(mg[:, 0])
        print(f"  merges: {len(mg)}  every {np.median(gap):.2f}s "
              f"(min {gap.min():.2f}, max {gap.max():.2f})")
        print(f"    overlap blended: median {np.median(mg[:, 2]):.0f} steps   "
              f"pure-new appended: median {np.median(mg[:, 3]):.0f}   "
              f"dropped (stale): median {np.median(mg[:, 4]):.0f}")
        if np.median(mg[:, 3]) < 2:
            print("    [!] almost no fresh horizon — chunks are consumed as fast as "
                  "they arrive; motion is running on blended/stale predictions")

    if "stream_cmd" not in data.files or len(data["stream_cmd"]) < 10:
        return

    st = data["stream_t"] - data["t"][0]
    sc = data["stream_cmd"]
    sdt = float(np.median(np.diff(st)))

    print(f"setpoint stream: {len(st)} samples @ {1.0 / sdt:.0f} Hz")
    _print_smoothness("stream_cmd", smoothness(sc, sdt))

    # Publish-TIMING jitter. The output filter smooths values, not the instants
    # they land — anything wrong here is invisible in stream_cmd and cannot be
    # fixed by tuning the filter.
    ivl = np.diff(st) * 1000.0
    nominal = 1000.0 / float(data["stream_hz"]) if "stream_hz" in data.files else sdt * 1000
    late = float((ivl > 1.5 * nominal).mean() * 100)
    print(f"  publish interval (ms): median {np.median(ivl):.2f}  "
          f"p95 {np.percentile(ivl, 95):.2f}  p99 {np.percentile(ivl, 99):.2f}  "
          f"max {ivl.max():.2f}  (nominal {nominal:.2f})")
    print(f"  intervals > 1.5x nominal: {late:.2f}%")
    if late > 1.0 or np.percentile(ivl, 99) > 1.5 * nominal:
        print("  [!] publish-timing jitter — the stream filter cannot fix this; "
              "suspect GIL contention from rclpy.spin / camera threads")
    if "stream_hz" in data.files and abs(float(data["stream_hz"]) - 125.0) < 1.0:
        print("  [!] stream_hz == controller_manager update_rate (125 Hz): the two "
              "clocks beat, so some control cycles get 0 and some 2 setpoints. "
              "moveit_servo oversamples at 200 Hz for exactly this reason — try "
              "--stream-hz 250")

    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    svel = np.diff(sc, axis=0) / sdt
    goal_t = data["t"] - data["t"][0]
    for j in range(6):
        axes3[0].plot(st, sc[:, j], lw=0.7, label=JOINT_NAMES[j])
        axes3[1].plot(st[1:], svel[:, j], lw=0.7)
    # The diagnostic view: with a lerp, velocity steps at every green line.
    for gt in goal_t:
        axes3[1].axvline(gt, color="green", lw=0.4, alpha=0.25)
    axes3[2].plot(st[1:], np.abs(np.diff(sc, axis=0)).max(axis=1), lw=0.7)
    axes3[0].set_ylabel("stream cmd (rad)")
    axes3[0].legend(loc="upper right", ncol=6, fontsize=16)
    axes3[1].set_ylabel("stream vel (rad/s)\n[green = control tick]")
    axes3[2].set_ylabel("max |Δcmd| per\nstream sample (rad)")
    axes3[2].set_xlabel("time (s)")
    axes3[0].set_title(cfg, fontsize=16)
    out3 = path.replace(".npz", "_stream.png")
    fig3.savefig(out3, dpi=120)
    print(f"Saved {out3}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "streaming_log.npz")
