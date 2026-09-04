"""Compare a simple_client chunk-mode log against a streaming-client log.

Usage:
    python scripts/compare_logs.py simple_client_log.npz streaming_log.npz \
        [--labels A B] [--joint elbow] [--move-thresh 0.02]

Answers one question: when the same policy is played at the same dt through the same
scaled_joint_trajectory_controller, is the streaming client's extra roughness in the
*trajectory it asks for* or in *how that trajectory is dispatched*?

Two stages are visible in both logs:
    cmd  — the commanded waypoint sequence
    js   — high-rate /joint_states, what the arm actually did

Three things make a naive comparison meaningless, and are handled here:

1. simple_client dwells. `send_chunk_action` blocks on the goal result plus a 1 s
   sleep, so its `js` is full of stationary stretches that deflate every pooled RMS.
   All metrics run on moving segments only (see --move-thresh).
2. simple_client logs the RAW chunk, before rts and before boundary blending; the
   streaming client logs post-filter, post-blend waypoints. Comparing them directly
   would credit simple_client with roughness it never sent. `--reconstruct` (default)
   replays rts + blend_chunk_boundary per chunk so both `cmd` series mean the same
   thing. Pass the same --chunk-filter-q/-r/--blend-steps the run used.
3. Different sample rates between `cmd` (1/dt) and `js` (driver rate): the tracking
   residual interpolates cmd onto the js timebase and removes the transport lag by
   cross-correlation first, so the residual is tracking error and not phase.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, welch

from filter_utils import blend_chunk_boundary, rts_smoother_chunk
from plot_streaming_log import JOINT_NAMES, smoothness

CFG_KEYS = ("control_interface", "segment_interp", "stream_filter", "jtc_end_velocity",
            "aggregate_fn_name", "dt", "stream_hz")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_log(path: str, args) -> dict:
    """Normalize either log format into one dict.

    Format sniff: only the streaming client writes `merge`/`prof`.
    """
    d = np.load(path, allow_pickle=False)
    if "cycles" in d.files:
        kind = "timed_chunk"
    elif "prof" in d.files or "merge" in d.files:
        kind = "streaming"
    else:
        kind = "simple"

    t = np.asarray(d["t"], dtype=float)
    cmd = np.asarray(d["cmd"], dtype=float)
    js = np.asarray(d["js"], dtype=float) if "js" in d.files else np.zeros((0, 7))
    chunk = np.asarray(d["chunk"]) if "chunk" in d.files else None

    dt = (float(d["dt"]) if "dt" in d.files
          else float(np.median(np.diff(t)[np.diff(t) > 0])))

    # Only simple_client logs the RAW chunk; timed_chunk logs what it dispatched, so
    # replaying rts + blend on it would apply both twice.
    stage = str(d["cmd_stage"]) if "cmd_stage" in d.files else (
        "raw" if kind == "simple" else "dispatched")
    if stage == "raw" and chunk is not None and args.reconstruct:
        cmd = _reconstruct_dispatched(cmd, chunk, dt, args)
        stage = "dispatched (reconstructed)"

    merge_t = (np.asarray(d["merge"], dtype=float)[:, 0]
               if "merge" in d.files and len(d["merge"]) else None)

    cfg = " ".join(f"{k}={d[k]}" for k in CFG_KEYS if k in d.files)
    if "chunk_filter" in d.files:
        cfg += f" chunk_filter={d['chunk_filter']}"

    return {
        "path": path, "kind": kind, "dt": dt, "cfg": cfg, "stage": stage,
        "cycles": np.asarray(d["cycles"], dtype=float) if "cycles" in d.files else None,
        "t": t, "cmd": cmd, "chunk": chunk,
        "js_t": js[:, 0] if len(js) else np.zeros(0),
        "js_p": js[:, 1:7] if len(js) else np.zeros((0, 6)),
        "merge_t": merge_t,
        "merge": np.asarray(d["merge"], dtype=float) if "merge" in d.files and len(d["merge"]) else None,
    }


def _reconstruct_dispatched(raw: np.ndarray, chunk: np.ndarray, dt: float, args) -> np.ndarray:
    """Replay simple_client's post-processing on the logged raw chunks.

    simple_client logs `arm_actions` straight from the policy, then applies
    rts_smoother_chunk and blend_chunk_boundary before dispatch. Both are
    deterministic given the raw chunk and the previous chunk's terminal state,
    so the dispatched trajectory is recoverable exactly.
    """
    out = np.empty_like(raw)
    prev_pos = prev_vel = None
    for c in np.unique(chunk):
        idx = np.flatnonzero(chunk == c)
        a = raw[idx]
        if args.chunk_filter == "rts" and len(a) >= 2:
            a = rts_smoother_chunk(a, dt=dt, q=args.chunk_filter_q, r=args.chunk_filter_r)
        if args.boundary_blend and prev_pos is not None:
            a = blend_chunk_boundary(a, prev_pos, prev_vel, dt, args.blend_steps)
        if len(a) >= 2:
            prev_pos, prev_vel = a[-1].copy(), (a[-1] - a[-2]) / dt
        out[idx] = a
    return out


# ---------------------------------------------------------------------------
# Moving-segment masking
# ---------------------------------------------------------------------------

def moving_runs(t: np.ndarray, pos: np.ndarray, thresh: float, min_seconds: float):
    """Index ranges [start, end) where the arm is actually moving.

    A boxcar over |v| keeps the mask from chattering at the threshold, which would
    otherwise shatter one motion into dozens of runs too short to differentiate.

    Each run is then trimmed by ~100 ms at both ends. The start and stop ramps are
    real acceleration, not roughness, and the reference run has one pair of them per
    chunk against the streaming run's one for the whole session — scoring them would
    penalize exactly the log that is supposed to be the smooth one.
    """
    if len(pos) < 12:
        return []
    dt_s = np.maximum(np.diff(t), 1e-9)
    # Thresholds in seconds, not samples: `js` arrives at 500 Hz from the driver but
    # the client drops most of it, so the realized rate varies by more than 10x
    # between logs. A sample count would silently reject every run in the slow one.
    min_samples = max(8, int(min_seconds / float(np.median(dt_s))))
    v = np.diff(pos, axis=0) / dt_s[:, None]
    speed = np.abs(v).max(axis=1)
    # ~150 ms boxcar regardless of sample rate: long enough to stop the mask
    # chattering at the threshold, short enough not to swallow a real pause.
    k = max(3, int(0.15 / float(np.median(dt_s))))
    speed = np.convolve(speed, np.ones(k) / k, mode="same")
    mask = speed > thresh

    trim = max(1, int(0.1 / float(np.median(dt_s))))
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_samples:
                runs.append((start + trim, i + 1 - trim))
            start = None
    if start is not None and len(mask) - start >= min_samples:
        runs.append((start + trim, len(mask) + 1 - trim))
    return [(a, b) for a, b in runs if b - a >= 8]


def pooled_smoothness(pos: np.ndarray, dt: float, runs) -> dict | None:
    """smoothness() per moving run, pooled: RMS in quadrature by length, max as max."""
    parts = [(b - a, smoothness(pos[a:b], dt)) for a, b in runs if b - a >= 8]
    if not parts:
        return None
    w = np.array([p[0] for p in parts], dtype=float)
    out = {}
    for key in parts[0][1]:
        vals = np.array([p[1][key] for p in parts], dtype=float)
        out[key] = (float(np.sqrt(np.sum(w * vals ** 2) / w.sum()))
                    if key.startswith("rms") else float(vals.max()))
    return out


# ---------------------------------------------------------------------------
# Spectrum and tracking
# ---------------------------------------------------------------------------

def velocity_psd(t: np.ndarray, pos: np.ndarray, runs):
    """Length-weighted average Welch PSD of joint velocity over the moving runs.

    Buzz has a frequency; a scalar jerk score does not. This is what separates
    "the republish rate is modulating the motion" from "the blending steps are"
    from "the timing is just noisy".
    """
    if not runs:
        return None, None
    fs = 1.0 / float(np.median(np.diff(t)))
    # One nperseg for every run, or the runs land on different frequency grids and
    # cannot be averaged. Runs shorter than it are dropped rather than zero-padded.
    lengths = [b - a - 1 for a, b in runs]
    nper = int(np.clip(np.percentile(lengths, 25), 32, 1024))
    if max(lengths) < nper:
        return None, None
    acc, wsum, freqs = None, 0.0, None
    for a, b in runs:
        seg = pos[a:b]
        if len(seg) - 1 < nper:
            continue
        v = np.diff(seg, axis=0) * fs
        f, p = welch(v, fs=fs, nperseg=nper, axis=0)
        p = p.sum(axis=1)                      # pool joints
        acc = p * len(v) if acc is None else acc + p * len(v)
        wsum += len(v)
        freqs = f
    if acc is None:
        return None, None
    return freqs, acc / wsum


def psd_peaks(f, p, fmin=0.5, n=3):
    if f is None:
        return []
    sel = f >= fmin
    fs_, ps_ = f[sel], p[sel]
    order = np.argsort(ps_)[::-1]
    picked = []
    for i in order:
        if all(abs(fs_[i] - fp) > 2.0 for fp, _ in picked):
            picked.append((float(fs_[i]), float(ps_[i])))
        if len(picked) == n:
            break
    return picked


def tracking(log: dict, runs, max_lag_s: float = 0.5):
    """Transport lag (ms) and RMS tracking residual after removing that lag.

    cmd is resampled onto the js timebase; the lag is estimated by cross-correlating
    the dominant joint's velocity, so the residual reports tracking error rather than
    the fixed delay every position-controlled arm has.
    """
    js_t, js_p, cmd, cmd_t = log["js_t"], log["js_p"], log["cmd"], log["t"]
    if not runs or len(js_t) < 10:
        return None
    ci = np.stack([np.interp(js_t, cmd_t, cmd[:, j]) for j in range(6)], axis=1)

    idx = np.concatenate([np.arange(a, min(b, len(js_p))) for a, b in runs])
    idx = idx[idx < len(js_p) - 1]
    if len(idx) < 64:
        return None

    fs = 1.0 / float(np.median(np.diff(js_t)))
    j = int(np.argmax(np.std(np.diff(js_p[idx], axis=0), axis=0)))
    a = np.diff(js_p[:, j])[idx]
    b = np.diff(ci[:, j])[idx]
    a = a - a.mean()
    b = b - b.mean()
    xc = correlate(a, b, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    lim = int(max_lag_s * fs)
    keep = np.abs(lags) <= lim
    lag = int(lags[keep][int(np.argmax(xc[keep]))])

    shifted = np.roll(ci, lag, axis=0)
    resid = js_p[idx] - shifted[idx]
    resid = resid - resid.mean(axis=0)          # drop constant offset, keep the wobble
    return {
        "lag_ms": lag / fs * 1000.0,
        "rms_resid": float(np.sqrt(np.mean(resid ** 2))),
        "max_resid": float(np.abs(resid).max()),
        "joint": JOINT_NAMES[j],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(m: dict | None) -> str:
    if m is None:
        return "  (no moving segments long enough)"
    return (f"rmsJ={m['rms_jerk']:10.1f}  maxJ={m['max_jerk']:11.1f}  "
            f"rmsA={m['rms_acc']:8.2f}  max|Δv|={m['max_dvel']:.4f}")


def analyze(log: dict, args) -> dict:
    runs = moving_runs(log["js_t"], log["js_p"], args.move_thresh, args.min_run)
    moving = sum(b - a for a, b in runs)
    total = max(1, len(log["js_p"]))

    print(f"\n=== {log['label']}  ({log['kind']}, cmd={log['stage']}, {log['path']}) ===")
    if log["cfg"]:
        print(f"  config: {log['cfg']}")
    js_hz = 1.0 / np.median(np.diff(log["js_t"])) if len(log["js_t"]) > 1 else float("nan")
    print(f"  dt={log['dt']:.4f}s  waypoints={len(log['cmd'])}  "
          f"js={len(log['js_p'])} samples @ {js_hz:.0f} Hz")
    print(f"  moving: {moving}/{total} js samples ({100.0 * moving / total:.0f}%) "
          f"in {len(runs)} run(s)")

    if np.isfinite(js_hz) and js_hz < 100.0:
        print(f"  [!] /joint_states logged at only {js_hz:.0f} Hz (driver publishes 500) "
              f"— Nyquist {js_hz/2:.0f} Hz, so the ~{1.0/log['dt']:.0f} Hz republish "
              f"artifact is at or past the fold and every measured-side number below "
              f"is suspect. rclpy.spin is losing messages to GIL contention.")

    cmd_runs = moving_runs(log["t"], log["cmd"], args.move_thresh, args.min_run)
    m_cmd = pooled_smoothness(log["cmd"], log["dt"], cmd_runs)
    m_js = pooled_smoothness(log["js_p"], float(np.median(np.diff(log["js_t"]))), runs) \
        if len(log["js_t"]) > 1 else None
    print(f"  cmd (commanded)  {_fmt(m_cmd)}")
    print(f"  js  (measured)   {_fmt(m_js)}")

    trk = tracking(log, runs)
    if trk:
        print(f"  tracking [{trk['joint']}]: lag {trk['lag_ms']:+.0f} ms   "
              f"rms residual {trk['rms_resid']*1000:.2f} mrad   "
              f"max {trk['max_resid']*1000:.2f} mrad")

    f, p = velocity_psd(log["js_t"], log["js_p"], runs)
    for i, (fp, pw) in enumerate(psd_peaks(f, p)):
        print(f"  velocity PSD peak {i+1}: {fp:7.2f} Hz   (power {pw:.3e})")

    if log["kind"] == "streaming" and len(log["t"]) > 2:
        ivl = np.diff(log["t"]) * 1000.0
        nom = log["dt"] * 1000.0
        print(f"  tick interval (ms): median {np.median(ivl):.1f}  "
              f"p95 {np.percentile(ivl, 95):.1f}  max {ivl.max():.1f}  (nominal {nom:.1f})")
        if np.percentile(ivl, 95) > 1.3 * nom:
            print("    [!] loop timing: every republish restarts time_from_start at "
                  "'now', so a late tick shifts the whole future trajectory")
    cy = log["cycles"]
    if cy is not None and len(cy) > 1:
        # (t, t_infer, s_obs, s_now, start_idx, lead, n_overlap, dispatched, scale, stale)
        ok = cy[cy[:, 9] == 0.0]
        gap = np.diff(cy[:, 0])
        print(f"  cycles: {len(cy)} every {np.median(gap):.2f}s   "
              f"infer median {np.median(cy[:, 1]):.2f}s p95 {np.percentile(cy[:, 1], 95):.2f}s")
        if len(ok):
            print(f"    lead median {np.median(ok[:, 5])*1000:.0f} ms   "
                  f"dispatched median {np.median(ok[:, 7]):.0f}   "
                  f"overlap median {np.median(ok[:, 6]):.0f}   "
                  f"speed scaling median {np.median(ok[:, 8]):.2f}")
        n_stale = int((cy[:, 9] == 1.0).sum())
        if n_stale:
            print(f"    [!] {n_stale} chunk(s) dropped as stale — inference outran the "
                  f"chunk, so the arm was holding at the end of the previous one")
        if len(ok) and np.median(ok[:, 8]) < 0.99:
            print(f"    [!] speed scaling below 1.0 — the arm is running slower than "
                  f"commanded; timing derived from dt alone would be wrong")

    if log["merge"] is not None and len(log["merge"]) > 1:
        mg = log["merge"]
        gap = np.diff(mg[:, 0])
        print(f"  merges: {len(mg)} every {np.median(gap):.2f}s  "
              f"(overlap blended median {np.median(mg[:, 2]):.0f}, "
              f"pure-new {np.median(mg[:, 3]):.0f}, dropped {np.median(mg[:, 4]):.0f})")
        if np.median(mg[:, 3]) < 2:
            print("    [!] almost no fresh horizon — running on blended predictions")

    log.update(runs=runs, psd=(f, p), m_cmd=m_cmd, m_js=m_js, trk=trk)
    return log


def make_figure(logs, args, out="compare.png"):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # Panel 1 — measured velocity of the joint that moves most in the first log
    ref = logs[0]
    j = (args.joint_index if args.joint_index is not None
         else int(np.argmax(np.std(np.diff(ref["js_p"], axis=0), axis=0))))
    for c, log in zip(colors, logs):
        jt, jp = log["js_t"], log["js_p"]
        if len(jt) < 2:
            continue
        v = np.diff(jp[:, j]) / np.diff(jt)
        axes[0].plot(jt[1:] - jt[0], v, lw=0.6, color=c, label=log["label"])
        if log["merge_t"] is not None:
            for mt in log["merge_t"]:
                axes[0].axvline(mt - jt[0], color=c, ls="--", lw=0.8, alpha=0.35)
        elif log["chunk"] is not None:
            bnd = log["t"][np.flatnonzero(np.diff(log["chunk"])) + 1]
            for bt in bnd:
                axes[0].axvline(bt - jt[0], color=c, ls=":", lw=0.8, alpha=0.35)
    axes[0].set_ylabel(f"measured {JOINT_NAMES[j]} vel (rad/s)")
    axes[0].set_xlabel("time (s)   [dashed = chunk merge, dotted = chunk boundary]")
    axes[0].legend(loc="upper right")

    # Panel 2 — velocity spectrum: where the buzz actually lives
    for c, log in zip(colors, logs):
        f, p = log["psd"]
        if f is None:
            continue
        axes[1].loglog(f[1:], p[1:], lw=0.8, color=c, label=log["label"])
    for log, c in zip(logs, colors):
        axes[1].axvline(1.0 / log["dt"], color=c, ls="--", lw=1.0, alpha=0.6)
        if log["merge_t"] is not None and len(log["merge_t"]) > 2:
            axes[1].axvline(1.0 / np.median(np.diff(log["merge_t"])), color=c,
                            ls=":", lw=1.0, alpha=0.6)
    axes[1].axvline(125.0, color="k", ls="-.", lw=0.8, alpha=0.5)
    axes[1].set_ylabel("joint-velocity PSD")
    axes[1].set_xlabel("Hz   [dashed = 1/dt (republish rate), dotted = merge rate, "
                       "dash-dot = 125 Hz controller]")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(loc="upper right")

    # Panel 3 — commanded roughness alone, controller excluded
    for c, log in zip(colors, logs):
        d = np.abs(np.diff(log["cmd"], axis=0)).max(axis=1)
        axes[2].plot(log["t"][1:] - log["t"][0], d, lw=0.8, color=c, label=log["label"])
    axes[2].set_ylabel("max |Δcmd| per waypoint (rad)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")

    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="npz logs (simple_client and/or streaming)")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--joint", default=None, help=f"one of {JOINT_NAMES}")
    ap.add_argument("--move-thresh", type=float, default=0.02,
                    help="rad/s; below this the arm counts as stationary")
    ap.add_argument("--min-run", type=float, default=0.5,
                    help="seconds; shorter moving runs are discarded")
    ap.add_argument("--out", default="compare.png")
    # simple_client post-processing replay (it logs the RAW chunk)
    ap.add_argument("--no-reconstruct", dest="reconstruct", action="store_false",
                    help="compare simple_client's raw policy output instead of what it dispatched")
    ap.add_argument("--chunk-filter", default="rts", choices=["none", "rts"])
    ap.add_argument("--chunk-filter-q", dest="chunk_filter_q", type=float, default=1e-3)
    ap.add_argument("--chunk-filter-r", dest="chunk_filter_r", type=float, default=1e-4)
    ap.add_argument("--no-boundary-blend", dest="boundary_blend", action="store_false")
    ap.add_argument("--blend-steps", type=int, default=4)
    args = ap.parse_args()
    args.joint_index = JOINT_NAMES.index(args.joint) if args.joint else None

    logs = []
    for i, path in enumerate(args.logs):
        log = load_log(path, args)
        log["label"] = (args.labels[i] if args.labels and i < len(args.labels)
                        else f"{chr(65+i)}:{path}")
        logs.append(analyze(log, args))

    if len({round(l["dt"], 4) for l in logs}) > 1:
        print("\n[!] logs have different dt — per-second jerk scales as 1/dt^3, so the "
              "comparison is dominated by playback speed, not smoothness")

    make_figure(logs, args, args.out)


if __name__ == "__main__":
    main()
