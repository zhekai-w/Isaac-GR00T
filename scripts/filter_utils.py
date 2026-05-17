import numpy as np
from scipy.signal import savgol_filter


class OneEuroFilter:
    """Adaptive low-pass filter that reduces lag at high speeds.

    At rest: heavy smoothing (cutoff = mincutoff).
    Moving fast: light smoothing (cutoff grows with speed via beta).
    """

    def __init__(self, freq: float, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x = None
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

    def __call__(self, x: float) -> float:
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) * self.freq
        a_d = self._alpha(self.dcutoff, self.freq)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.mincutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, self.freq)
        self._x = a * x + (1 - a) * self._x
        return self._x


def savgol_chunk(arm_chunk: np.ndarray, window_length: int = 7, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smooth over time axis of arm trajectory chunk (H, N)."""
    return savgol_filter(arm_chunk, window_length, polyorder, axis=0)


def rts_smoother_chunk(arm_chunk: np.ndarray, dt: float = 0.15,
                        q: float = 1e-3, r: float = 1e-4) -> np.ndarray:
    """RTS (Rauch-Tung-Striebel) Kalman smoother on arm trajectory chunk.

    Constant-velocity state model per joint: state = [position, velocity].
    q: process noise — larger = more responsive, less smooth.
    r: measurement noise — larger = more smoothing.
    arm_chunk: (H, N) — works for any number of joints N.
    Returns (H, N) smoothed positions.
    """
    H, n_joints = arm_chunk.shape
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Hm = np.array([[1.0, 0.0]])
    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    R = np.array([[r]])

    smoothed = np.zeros_like(arm_chunk)
    for j in range(n_joints):
        z = arm_chunk[:, j]
        # Forward Kalman pass
        xs = np.zeros((H, 2))
        Ps = np.zeros((H, 2, 2))
        x = np.array([z[0], 0.0])
        P = np.eye(2)
        for t in range(H):
            x = F @ x
            P = F @ P @ F.T + Q
            S = (Hm @ P @ Hm.T + R)[0, 0]
            K = (P @ Hm.T) / S
            x = x + K.flatten() * (z[t] - (Hm @ x)[0])
            P = (np.eye(2) - K @ Hm) @ P
            xs[t], Ps[t] = x, P
        # Backward RTS smoother pass
        sm = xs.copy()
        Psm = Ps.copy()
        for t in range(H - 2, -1, -1):
            P_pred = F @ Ps[t] @ F.T + Q
            G = Ps[t] @ F.T @ np.linalg.inv(P_pred)
            sm[t] = xs[t] + G @ (sm[t + 1] - F @ xs[t])
            Psm[t] = Ps[t] + G @ (Psm[t + 1] - P_pred) @ G.T
        smoothed[:, j] = sm[:, 0]
    return smoothed


def blend_chunk_boundary(arm_chunk: np.ndarray, prev_end_pos: np.ndarray,
                          prev_end_vel: np.ndarray, dt: float,
                          blend_steps: int = 4) -> np.ndarray:
    """Cosine-ramp first blend_steps waypoints from previous chunk's terminal state.

    Eliminates hard positional jumps at chunk boundaries.
    prev_end_pos: (N,) last commanded position of previous chunk.
    prev_end_vel: (N,) estimated velocity at end of previous chunk (units/s).
    """
    blended = arm_chunk.copy()
    n = min(blend_steps, arm_chunk.shape[0])
    for step in range(n):
        alpha = 0.5 * (1.0 - np.cos(np.pi * (step + 1) / n))  # 0 → 1
        predicted = prev_end_pos + prev_end_vel * dt * (step + 1)
        blended[step] = (1.0 - alpha) * predicted + alpha * arm_chunk[step]
    return blended
