# -*- coding: utf-8 -*-
"""
公共模块：阵列几何、信号仿真、SRP-PHAT 特征、CRB、评估指标。

与原代码（条带 + 两点之间搜索 / CCFRC_BTP）保持一致的部分：
  - 8 元均匀圆阵，直径 88.75 mm，fs = 50 kHz，0.5 s 线性 chirp（500–2500 Hz）
  - 源信号 = chirp + 0.1 * 白噪声（各麦克风共享），每路再加独立噪声（默认 std = 0.6）
  - STFT：Hamming 窗 20 ms，hop 10 ms，n_fft = 1024，center 补零
  - 先对所有帧平均互谱，再做 PHAT 归一化，然后做频域 SRP（与原 srp_score_map 等价）

相对原代码的修改（都可以通过参数切换回原样，便于做消融）：
  1. 分数时延：原代码用 .long() 把时延截断为整数采样点，最大时延只有约 13 个采样点，
     截断会引入几度的系统误差。这里默认用频域相移实现精确的分数时延，
     delay_mode="integer" 可复现原做法。
  2. 噪声加在整段信号上（原代码只加在时延之后的部分）。
  3. 原代码先把协方差在帧上取平均，再 repeat 回每一帧逐帧计算 SRP，最后再取平均，
     同一个结果重复算了约 50 次。这里只算一次，结果完全相同，但快得多。
  4. 只用 i<j 的麦克风对（原代码 triu_indices 包含 i=j 的自相关项，只是一个常数偏移）。
  5. 打分有两种后端，结果等价：
       - "freq"：频域逐频点求和，和原代码完全一致，较慢；
       - "lut" ：先把每对麦克风的 PHAT 互谱 IFFT 成 GCC（16 倍上采样），
                 再按时延查表求和（论文式 (2) 的形式），很快。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

C_SOUND = 343.0
FS = 50000
DURATION = 0.5
MIC_RADIUS = 0.08875 / 2.0  # 44.375 mm
N_MICS = 8


# ======================================================================
# 几何与坐标
# ======================================================================
def make_uca(n_mics: int = N_MICS, radius: float = MIC_RADIUS) -> np.ndarray:
    """均匀圆阵，麦克风在 z = 0 平面上。返回 (M, 3)。"""
    alpha = 2.0 * np.pi * np.arange(n_mics) / n_mics
    return np.stack([radius * np.cos(alpha), radius * np.sin(alpha), np.zeros(n_mics)], axis=1)


def dir_vec(az_deg, el_deg) -> np.ndarray:
    """方位角/俯仰角（度）→ 单位向量。俯仰角从水平面向上量。支持数组广播。"""
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    return np.stack([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)], axis=-1)


def vec_to_azel(v: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(v, dtype=float)
    az = np.degrees(np.arctan2(v[1], v[0]))
    el = np.degrees(np.arcsin(np.clip(v[2] / (np.linalg.norm(v) + 1e-12), -1.0, 1.0)))
    return float(az), float(np.clip(el, 0.0, 90.0))


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def az_error(est, true):
    return np.abs(wrap180(np.asarray(est) - np.asarray(true)))


def parabolic_vertex(y_m1: float, y0: float, y_p1: float) -> float:
    """三点抛物线插值，返回顶点相对中心点的偏移（以步长为单位，范围 [-0.5, 0.5]）。"""
    denom = y_m1 - 2.0 * y0 + y_p1
    if denom >= -1e-12:  # 不是向下开口
        return 0.0
    return float(np.clip(0.5 * (y_m1 - y_p1) / denom, -0.5, 0.5))


# ======================================================================
# 信号仿真
# ======================================================================
def make_source(fs: int = FS, duration: float = DURATION, f0: float = 500.0, f1: float = 2500.0,
                shared_noise_std: float = 0.1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """与原代码一致：线性 chirp + 共享白噪声（这部分噪声对所有麦克风是相干的，属于"信号"）。"""
    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(int(round(fs * duration))) / fs
    chirp = np.sin(2.0 * np.pi * (f0 * t + (f1 - f0) / (2.0 * duration) * t ** 2))
    return chirp + shared_noise_std * rng.standard_normal(len(t))


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """用频域相移实现分数时延（补零避免循环卷绕），输出长度与输入相同。"""
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, nfft)
    k = np.arange(len(X))
    y = np.fft.irfft(X * np.exp(-2j * np.pi * k * delay_samples / nfft), nfft)
    return y[:n]


def noise_std_from_snr(source: np.ndarray, snr_db: float) -> float:
    return float(np.sqrt(np.mean(source ** 2) / 10.0 ** (snr_db / 10.0)))


def snr_from_noise_std(source: np.ndarray, noise_std: float) -> float:
    return float(10.0 * np.log10(np.mean(source ** 2) / noise_std ** 2))


def simulate_freefield(az_deg: float, el_deg: float, mics: np.ndarray, source: np.ndarray,
                       noise_std: float = 0.6, src_dist: float = 3.0, fs: int = FS,
                       delay_mode: str = "fractional",
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """自由场仿真（与原代码相同：点声源在 3 m 处，球面波传播时延，不考虑幅度衰减）。

    delay_mode: "fractional"（默认，精确）或 "integer"（复现原代码的 .long() 截断）。
    返回 (M, N) 麦克风信号。
    """
    rng = np.random.default_rng() if rng is None else rng
    src = src_dist * dir_vec(az_deg, el_deg)
    delays = np.linalg.norm(src[None, :] - mics, axis=1) / C_SOUND * fs
    n = len(source)
    out = np.zeros((mics.shape[0], n))
    for m, d in enumerate(delays):
        if delay_mode == "integer":
            di = int(d)  # 与原代码 .long() 一致：向下截断
            out[m, di:] = source[: n - di]
        else:
            out[m] = fractional_delay(source, d)
        out[m] += noise_std * rng.standard_normal(n)
    return out


def simulate_room(az_deg: float, el_deg: float, mics_local: np.ndarray, source: np.ndarray,
                  rt60: float, noise_std: float, room_dim=(7.0, 6.0, 3.2),
                  array_center=(3.5, 3.0, 1.2), src_dist: float = 2.0, fs: int = FS,
                  rng: Optional[np.random.Generator] = None) -> Optional[np.ndarray]:
    """用 pyroomacoustics（镜像源法）仿真混响房间。声源不在房间内时返回 None。"""
    import pyroomacoustics as pra

    rng = np.random.default_rng() if rng is None else rng
    room_dim = np.asarray(room_dim, dtype=float)
    center = np.asarray(array_center, dtype=float)
    src = center + src_dist * dir_vec(az_deg, el_deg)
    if np.any(src < 0.3) or np.any(src > room_dim - 0.3):
        return None
    e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_abs), max_order=max_order)
    room.add_source(src, signal=source)
    room.add_microphone_array((center[None, :] + mics_local).T)
    room.simulate()
    sig = room.mic_array.signals[:, : len(source)]
    if sig.shape[1] < len(source):
        sig = np.pad(sig, ((0, 0), (0, len(source) - sig.shape[1])))
    return sig + noise_std * rng.standard_normal(sig.shape)


# ======================================================================
# SRP-PHAT 特征（互谱 → PHAT → GCC 查找表）
# ======================================================================
def _hamming_periodic(n: int) -> np.ndarray:
    # 与 torch.hamming_window(n)（periodic=True）一致
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / n)


def stft(x: np.ndarray, fs: int = FS, win_ms: float = 20.0, hop_ms: float = 10.0,
         n_fft: int = 1024) -> np.ndarray:
    """与 torch.stft(center=True, pad_mode='constant') 的帧划分一致。返回 (M, T, K)。"""
    win_len = int(round(fs * win_ms / 1000.0))
    hop = int(round(fs * hop_ms / 1000.0))
    w = np.zeros(n_fft)
    off = (n_fft - win_len) // 2
    w[off: off + win_len] = _hamming_periodic(win_len)
    xp = np.pad(x, ((0, 0), (n_fft // 2, n_fft // 2)))
    n_frames = 1 + (xp.shape[1] - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = xp[:, idx] * w  # (M, T, n_fft)
    return np.fft.rfft(frames, axis=-1)


@dataclass
class SRPFeatures:
    """一次观测的 SRP-PHAT 特征。所有方法共用同一份特征，保证对比公平。"""
    mics: np.ndarray
    fs: int
    n_fft: int
    pairs: np.ndarray          # (P, 2)
    cross: np.ndarray          # (P, K) PHAT 归一化后的平均互谱
    lut: np.ndarray            # (P, 2L+1) 上采样 GCC
    lut_L: int
    upsample: int
    baselines: np.ndarray      # (P, 3) m_i - m_j
    n_eval: int = 0            # 打分次数计数器（与硬件无关的代价指标）

    # ---------- 构造 ----------
    @staticmethod
    def from_signals(sig: np.ndarray, mics: np.ndarray, fs: int = FS, n_fft: int = 1024,
                     win_ms: float = 20.0, hop_ms: float = 10.0, upsample: int = 16,
                     band_hz: Optional[Tuple[float, float]] = None) -> "SRPFeatures":
        X = stft(sig, fs=fs, win_ms=win_ms, hop_ms=hop_ms, n_fft=n_fft)  # (M, T, K)
        M = mics.shape[0]
        pairs = np.array([(i, j) for i in range(M) for j in range(i + 1, M)])
        cross = np.mean(X[pairs[:, 0]] * np.conj(X[pairs[:, 1]]), axis=1)  # 先对帧平均（同原代码）
        cross = cross / (np.abs(cross) + 1e-20)                              # 再 PHAT 归一化
        if band_hz is not None:
            f = np.arange(cross.shape[1]) * fs / n_fft
            cross = cross * ((f >= band_hz[0]) & (f <= band_hz[1]))[None, :]
        baselines = mics[pairs[:, 0]] - mics[pairs[:, 1]]
        max_lag = np.max(np.linalg.norm(baselines, axis=1)) / C_SOUND * fs
        L = int(np.ceil(max_lag * upsample)) + 4
        n_up = n_fft * upsample
        # irfft 补零 = 时域带限插值。乘 upsample 使完全相干时峰值约为 1。
        r = np.fft.irfft(cross, n=n_up, axis=1) * upsample
        lut = np.concatenate([r[:, n_up - L:], r[:, : L + 1]], axis=1)  # 时延 -L..L（单位 1/upsample 采样点）
        return SRPFeatures(mics=mics, fs=fs, n_fft=n_fft, pairs=pairs, cross=cross, lut=lut,
                           lut_L=L, upsample=upsample, baselines=baselines)

    # ---------- 每对麦克风的理论时延（采样点）----------
    def pair_delays(self, dirs: np.ndarray) -> np.ndarray:
        """远场模型：到达时间差 t_i - t_j = -(m_i - m_j)·u / c。dirs: (D,3) → (D,P)。"""
        return -(dirs @ self.baselines.T) * self.fs / C_SOUND

    # ---------- 打分 ----------
    def score_dirs(self, dirs: np.ndarray, backend: str = "lut") -> np.ndarray:
        dirs = np.atleast_2d(dirs)
        self.n_eval += dirs.shape[0]
        if backend == "freq":
            return self._score_freq(dirs)
        return self._score_lut(dirs)

    def score_azel(self, az_deg, el_deg, backend: str = "lut") -> np.ndarray:
        az_deg, el_deg = np.broadcast_arrays(np.asarray(az_deg, float), np.asarray(el_deg, float))
        s = self.score_dirs(dir_vec(az_deg.ravel(), el_deg.ravel()), backend=backend)
        return s.reshape(az_deg.shape)

    def _score_lut(self, dirs: np.ndarray) -> np.ndarray:
        pos = self.pair_delays(dirs) * self.upsample + self.lut_L  # (D, P)
        i0 = np.clip(np.floor(pos).astype(int), 0, self.lut.shape[1] - 2)
        fr = pos - i0
        p = np.arange(self.lut.shape[0])[None, :]
        v = (1.0 - fr) * self.lut[p, i0] + fr * self.lut[p, i0 + 1]
        return v.mean(axis=1)

    def _score_freq(self, dirs: np.ndarray, chunk: int = 128) -> np.ndarray:
        K = self.cross.shape[1]
        wk = np.full(K, 2.0)
        wk[0] = 1.0
        wk[-1] = 1.0
        omega = 2.0 * np.pi * np.arange(K) / self.n_fft
        out = np.empty(dirs.shape[0])
        for s in range(0, dirs.shape[0], chunk):
            tau = self.pair_delays(dirs[s: s + chunk])  # (d, P)
            ph = np.exp(1j * tau[:, :, None] * omega[None, None, :])
            val = np.real(np.einsum("pk,dpk->dp", self.cross * wk, ph)) / self.n_fft
            out[s: s + chunk] = val.mean(axis=1)
        return out

    # ---------- GCC 峰值（给 TDOA 方法用）----------
    def gcc_peak_delays(self, margin_samples: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """每对麦克风在物理允许范围内找 GCC-PHAT 峰值，返回 (时延[采样点], 峰值高度)。"""
        L, U = self.lut_L, self.upsample
        lags = (np.arange(self.lut.shape[1]) - L) / U
        maxlag = np.linalg.norm(self.baselines, axis=1) / C_SOUND * self.fs + margin_samples
        tau = np.empty(len(self.pairs))
        height = np.empty(len(self.pairs))
        for p in range(len(self.pairs)):
            valid = np.abs(lags) <= maxlag[p]
            idx = np.flatnonzero(valid)
            k = idx[np.argmax(self.lut[p, idx])]
            off = 0.0
            if idx[0] < k < idx[-1]:
                off = parabolic_vertex(self.lut[p, k - 1], self.lut[p, k], self.lut[p, k + 1])
            tau[p] = lags[k] + off / U
            height[p] = self.lut[p, k]
        return tau, height


# ======================================================================
# 克拉美-罗下界（远场、确定性未知波形、白噪声）
# ======================================================================
def crb_std_deg(az_deg: float, el_deg: float, mics: np.ndarray, source: np.ndarray,
                noise_std: float, fs: int = FS) -> Tuple[float, float]:
    """返回 (方位角 CRB 标准差, 俯仰角 CRB 标准差)，单位：度。

    y_m(n) = s(n - d_m) + w_m(n)，w ~ N(0, σ²)。对实信号：
        J = (E_ṡ / σ²) · Gᵀ G，  G[m, :] = ∂d_m/∂(φ, θ)（秒），并去掉 m 上的均值（未知波形）
    其中 E_ṡ = Σ_n ṡ(t_n)²。这里用的是实际仿真的源信号（含共享噪声那部分宽带成分）。
    """
    N = len(source)
    S = np.fft.rfft(source)
    w = 2.0 * np.pi * np.fft.rfftfreq(N, 1.0 / fs)
    wk = np.full(len(S), 2.0)
    wk[0] = 1.0
    if N % 2 == 0:
        wk[-1] = 1.0
    E_sdot = float(np.sum(wk * np.abs(w * S) ** 2) / N)

    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    du_daz = np.array([-np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), 0.0])
    du_del = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    G = np.stack([-(mics @ du_daz) / C_SOUND, -(mics @ du_del) / C_SOUND], axis=1)  # (M, 2) 秒
    G = G - G.mean(axis=0, keepdims=True)
    J = (E_sdot / noise_std ** 2) * (G.T @ G)
    try:
        crb = np.linalg.inv(J)
        sd = np.sqrt(np.clip(np.diag(crb), 0, None))
    except np.linalg.LinAlgError:
        sd = np.array([np.inf, np.inf])
    return float(np.degrees(sd[0])), float(np.degrees(sd[1]))


# ======================================================================
# 评估工具
# ======================================================================
def sample_directions(n: int, rng: np.random.Generator, el_max: float = 89.0,
                      mode: str = "uniform_angle") -> np.ndarray:
    """随机方向（不在网格上）。
    uniform_angle：俯仰角在 [0, el_max] 上均匀，便于看每个俯仰区间的误差；
    uniform_sphere：在上半球面均匀（俯仰角密度 ∝ cosθ）。
    返回 (n, 2) 的 [方位角, 俯仰角]（度）。"""
    az = rng.uniform(-180.0, 180.0, n)
    if mode == "uniform_sphere":
        el = np.degrees(np.arcsin(rng.uniform(0.0, np.sin(np.deg2rad(el_max)), n)))
    else:
        el = rng.uniform(0.0, el_max, n)
    return np.stack([az, el], axis=1)


def summarize(az_err: np.ndarray, el_err: np.ndarray, true_el: np.ndarray,
              az_excl_el: float = 85.0, outlier_deg: float = 10.0) -> Dict[str, float]:
    keep = true_el < az_excl_el
    ae, ee = az_err[keep], el_err
    return {
        "az_rmse": float(np.sqrt(np.mean(ae ** 2))),
        "az_median": float(np.median(ae)),
        "az_outlier_rate": float(np.mean(ae > outlier_deg)),
        "el_rmse": float(np.sqrt(np.mean(ee ** 2))),
        "el_median": float(np.median(ee)),
        "el_outlier_rate": float(np.mean(ee > outlier_deg)),
        "n": int(len(el_err)),
    }


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0
