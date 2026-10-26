# -*- coding: utf-8 -*-
"""
方法三：TDOA + 贝叶斯（不做方向搜索）。

第一步：用 GCC-PHAT 测出每对麦克风的时间差 τ̂（在物理允许范围内找峰，上采样 + 抛物线插值）。
        平面阵列的时间差只依赖方向在桌面上的投影 q = (cosθcosφ, cosθsinφ)：
            τ = −(fs/c) · B q + ε，   B 的每一行是一对麦克风的基线 (m_i − m_j) 的 x、y 分量
        用加权最小二乘一步解出 q̂。

时延误差的协方差 Σ_τ：
  28 对麦克风只有 7 个独立时间差，共用同一个麦克风的两对，误差是相关的。这里用
      Σ_τ = σ² (D Dᵀ + λ I)
  D 是"麦克风对 × 麦克风"的差分矩阵，D Dᵀ 描述"每个麦克风自带的时延误差"带来的相关性，
  λ I 描述每对麦克风各自独立的误差（也保证矩阵可逆）。σ² 从本次拟合残差估计。
  robust=True 时，会迭代剔除残差过大的麦克风对（GCC 挑错峰的离群值）。

第二步：Σ_q = (c/fs)² σ² (Bᵀ W B)⁻¹，W = (D Dᵀ + λI)⁻¹。

第三步：在 (φ, θ) 网格上构造后验
      p(φ, θ | q̂) ∝ exp[ −½ (q(φ,θ) − q̂)ᵀ Σ_q⁻¹ (q(φ,θ) − q̂) ] · p(φ, θ)
  输出：
    - ML    ：φ = atan2(q̂_y, q̂_x)，θ = arccos(min(|q̂|, 1))（|q̂| > 1 时只能截断到 0°）
    - MAP   ：后验最大点
    - MMSE  ：俯仰角后验均值 E[θ | q̂]，方位角用后验的圆周均值

前提假设：远场、单声源、时延误差近似高斯。混响和低信噪比下 GCC 容易挑错峰，
此时误差不再是高斯的，算出来的 Σ_q 和后验都会过于"自信"，这正是需要实验检验的地方。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from common import C_SOUND, SRPFeatures, dir_vec


def _diff_matrix(pairs: np.ndarray, n_mics: int) -> np.ndarray:
    D = np.zeros((len(pairs), n_mics))
    D[np.arange(len(pairs)), pairs[:, 0]] = 1.0
    D[np.arange(len(pairs)), pairs[:, 1]] = -1.0
    return D


def tdoa_ls(feat: SRPFeatures, lam: float = 0.2, robust: bool = True, sigma_floor: float = 0.02,
            n_iter: int = 3, thresh: float = 3.0) -> Dict:
    """GCC-PHAT 时延 + 加权最小二乘求 q̂。时延单位：采样点。"""
    tau, height = feat.gcc_peak_delays()
    B = -(feat.fs / C_SOUND) * feat.baselines[:, :2]           # τ = B q
    D = _diff_matrix(feat.pairs, feat.mics.shape[0])
    S0 = D @ D.T + lam * np.eye(len(tau))                      # Σ_τ / σ²
    keep = np.ones(len(tau), dtype=bool)
    for _ in range(n_iter if robust else 1):
        W = np.linalg.inv(S0[np.ix_(keep, keep)])
        Bk, tk = B[keep], tau[keep]
        A = Bk.T @ W @ Bk
        q = np.linalg.solve(A, Bk.T @ W @ tk)
        r = tk - Bk @ q
        dof = max(int(keep.sum()) - 2, 1)
        sigma2 = max(float(r @ W @ r) / dof, sigma_floor ** 2)
        if not robust:
            break
        r_all = tau - B @ q
        z = np.abs(r_all) / np.sqrt(sigma2 * np.diag(S0))
        new_keep = z < thresh
        if new_keep.sum() < 10 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    Sigma_q = sigma2 * np.linalg.inv(A)
    return {"q": q, "Sigma_q": Sigma_q, "sigma_tau": float(np.sqrt(sigma2)),
            "n_pairs_used": int(keep.sum()), "tau": tau, "gcc_height": height}


def _log_prior(theta_deg: np.ndarray, prior: str) -> np.ndarray:
    if prior == "sphere":
        return np.log(np.clip(np.cos(np.deg2rad(theta_deg)), 1e-6, None))
    return np.zeros_like(theta_deg)


def tdoa_bayes(feat: SRPFeatures, prior: str = "flat", az_step: float = 1.0, el_step: float = 0.25,
               **ls_kw) -> Dict[str, Tuple[float, float]]:
    """返回三种估计：{"ml": (az, el), "map": (az, el), "mmse": (az, el)}，以及诊断信息。"""
    feat.n_eval = 0
    ls = tdoa_ls(feat, **ls_kw)
    q, Sq = ls["q"], ls["Sigma_q"]

    # ML（直接反解）
    az_ml = float(np.degrees(np.arctan2(q[1], q[0])))
    el_ml = float(np.degrees(np.arccos(np.clip(np.linalg.norm(q), 0.0, 1.0))))

    # 后验（q 空间的高斯似然 × 先验）
    az = np.arange(-180.0, 180.0, az_step)
    el = np.arange(0.0, 90.0 + 1e-9, el_step)
    A, E = np.meshgrid(az, el)                                     # (Nel, Naz)
    Q = dir_vec(A, E)[..., :2]                                     # (Nel, Naz, 2)
    d = Q - q
    Si = np.linalg.inv(Sq)
    m2 = np.einsum("...i,ij,...j->...", d, Si, d)
    logp = -0.5 * m2 + _log_prior(E, prior)
    w = np.exp(logp - logp.max())
    w /= w.sum()

    ie, ia = np.unravel_index(int(np.argmax(w)), w.shape)
    az_map, el_map = float(az[ia]), float(el[ie])

    w_el = w.sum(axis=1)
    el_mmse = float(np.sum(w_el * el))
    el_std = float(np.sqrt(np.sum(w_el * (el - el_mmse) ** 2)))
    w_az = w.sum(axis=0)
    az_mmse = float(np.degrees(np.arctan2(np.sum(w_az * np.sin(np.deg2rad(az))),
                                          np.sum(w_az * np.cos(np.deg2rad(az))))))
    cdf = np.cumsum(w_el)
    ci = (float(el[np.searchsorted(cdf, 0.05)]), float(el[min(np.searchsorted(cdf, 0.95), len(el) - 1)]))
    return {"ml": (az_ml, el_ml), "map": (az_map, el_map), "mmse": (az_mmse, el_mmse),
            "diag": {"q_norm": float(np.linalg.norm(q)), "sigma_tau_samples": ls["sigma_tau"],
                     "n_pairs_used": ls["n_pairs_used"], "el_post_std": el_std,
                     "el_ci90": ci, "n_eval": 0}}


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from common import make_uca, make_source, simulate_freefield, az_error

    rng = np.random.default_rng(2)
    mics = make_uca()
    print(f"{'真值 az/el':>14} | {'ML':>14} | {'MMSE':>14} | |q̂|   σ_τ(采样点) 用到的对数 | 俯仰 90% 区间")
    for _ in range(8):
        az, el = rng.uniform(-180, 180), rng.uniform(0, 89)
        x = simulate_freefield(az, el, mics, make_source(rng=rng), rng=rng)
        f = SRPFeatures.from_signals(x, mics)
        r = tdoa_bayes(f)
        d = r["diag"]
        print(f"{az:7.1f} {el:5.1f} | {r['ml'][0]:7.1f} {r['ml'][1]:5.1f} | {r['mmse'][0]:7.1f} {r['mmse'][1]:5.1f} | "
              f"{d['q_norm']:.3f}  {d['sigma_tau_samples']:.3f}       {d['n_pairs_used']:2d}       | "
              f"[{d['el_ci90'][0]:.1f}, {d['el_ci90'][1]:.1f}]")
