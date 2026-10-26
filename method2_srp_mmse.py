# -*- coding: utf-8 -*-
"""
方法二：SRP 后验均值（Stage 1 与 ASAP 相同，Stage 2 改成 MMSE）。

思路：把固定方位 φ̂ 上的 SRP 得分曲线当作"似然"，

    p(θ | 数据) ∝ exp( β · [P(φ̂, θ) − max P] ) · p(θ)

输出后验均值 E[θ | 数据]（最小均方误差估计），同时给出后验标准差和 90% 置信区间。

  - β（温度）：控制把 SRP 得分"看得多尖"。β 越大越接近取峰值；β 越小越平滑、越依赖先验。
    β 不能拍脑袋定，这里用 calibrate_beta() 在**单独的一组校准数据**上选（不要用测试数据选）。
  - 先验 p(θ)：
      "flat"   —— 在俯仰角上均匀；
      "sphere" —— 在上半球面上均匀方向，对应 p(θ) ∝ cosθ（"没有偏好"其实也是一种先验）。

注意：MMSE 保证的是在先验分布下的**平均**平方误差最小，不保证每个真实角度都更准；
数据越弱的地方（水平面附近的俯仰角），先验和 β 的影响越大，结果会被往先验中心拉。
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from common import SRPFeatures
from method1_asap_peak import stage1_azimuth


def log_prior(theta_deg: np.ndarray, prior: str) -> np.ndarray:
    if prior == "sphere":
        return np.log(np.clip(np.cos(np.deg2rad(theta_deg)), 1e-6, None))
    return np.zeros_like(theta_deg)


def posterior_from_scores(theta: np.ndarray, P: np.ndarray, beta: float,
                          prior: str = "flat") -> Dict[str, float]:
    logw = beta * (P - P.max()) + log_prior(theta, prior)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    mean = float(np.sum(w * theta))
    std = float(np.sqrt(np.sum(w * (theta - mean) ** 2)))
    cdf = np.cumsum(w)
    lo = float(theta[np.searchsorted(cdf, 0.05)])
    hi = float(theta[min(np.searchsorted(cdf, 0.95), len(theta) - 1)])
    return {"mean": mean, "std": std, "ci90_lo": lo, "ci90_hi": hi,
            "map": float(theta[int(np.argmax(logw))])}


def srp_mmse(feat: SRPFeatures, beta: float, prior: str = "flat", el_step: float = 1.0,
             el_window: Optional[float] = None, **stage1_kw) -> Tuple[float, float, Dict]:
    """返回 (方位角, 俯仰角后验均值, 诊断信息)。

    el_window=None：在整个 [0°, 90°] 上算后验（最稳妥，91 次打分）；
    给定数值 r：只在 Stage 1 粗俯仰 ± r 内算（打分更少，但窗口外的概率被忽略）。
    """
    feat.n_eval = 0
    phi_hat, th_hat, _ = stage1_azimuth(feat, **stage1_kw)
    n1 = feat.n_eval
    if el_window is None:
        theta = np.arange(0.0, 90.0 + 1e-9, el_step)
    else:
        theta = np.arange(max(0.0, th_hat - el_window), min(90.0, th_hat + el_window) + 1e-9, el_step)
    P = feat.score_azel(phi_hat, theta)
    post = posterior_from_scores(theta, P, beta, prior)
    return phi_hat, post["mean"], {"n_eval": feat.n_eval, "n_eval_stage1": n1,
                                   "stage1_el": th_hat, **{f"post_{k}": v for k, v in post.items()}}


def calibrate_beta(feats: Sequence[SRPFeatures], true_el: Sequence[float],
                   betas: Iterable[float] = (10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240),
                   prior: str = "flat", el_step: float = 1.0) -> Tuple[float, Dict[float, float]]:
    """在校准数据上选 β：使俯仰角 RMSE 最小。

    为了省时间，每个样本只做一次 Stage 1 和一次俯仰扫描，然后对所有 β 重复使用同一条得分曲线。
    返回 (最优 β, {β: RMSE})。
    """
    betas = list(betas)
    theta = np.arange(0.0, 90.0 + 1e-9, el_step)
    curves = []
    for f in feats:
        phi_hat, _, _ = stage1_azimuth(f)
        curves.append(f.score_azel(phi_hat, theta))
    true_el = np.asarray(true_el, dtype=float)
    table = {}
    for b in betas:
        est = np.array([posterior_from_scores(theta, P, b, prior)["mean"] for P in curves])
        table[b] = float(np.sqrt(np.mean((est - true_el) ** 2)))
    best = min(table, key=table.get)
    return best, table


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from common import make_uca, make_source, simulate_freefield
    from method1_asap_peak import asap_peak

    rng = np.random.default_rng(1)
    mics = make_uca()

    # 1) 用一组单独的校准数据选 β
    cal_feats, cal_el = [], []
    for _ in range(40):
        az, el = rng.uniform(-180, 180), rng.uniform(0, 89)
        x = simulate_freefield(az, el, mics, make_source(rng=rng), rng=rng)
        cal_feats.append(SRPFeatures.from_signals(x, mics))
        cal_el.append(el)
    beta, table = calibrate_beta(cal_feats, cal_el)
    print("β 校准（俯仰 RMSE）：", {k: round(v, 3) for k, v in table.items()}, "→ 选 β =", beta)

    # 2) 在新的测试方向上对比取峰值和后验均值
    print(f"\n{'真值 el':>8} | {'ASAP 峰值':>9} | {'后验均值':>8} | {'90% 区间':>14}")
    for _ in range(8):
        az, el = rng.uniform(-180, 180), rng.uniform(0, 20)   # 重点看水平面附近
        x = simulate_freefield(az, el, mics, make_source(rng=rng), rng=rng)
        f = SRPFeatures.from_signals(x, mics)
        _, el_pk, _ = asap_peak(f)
        _, el_mm, d = srp_mmse(f, beta)
        print(f"{el:8.2f} | {el_pk:9.2f} | {el_mm:8.2f} | [{d['post_ci90_lo']:5.1f}, {d['post_ci90_hi']:5.1f}]")
