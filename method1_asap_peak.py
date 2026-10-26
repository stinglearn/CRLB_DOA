# -*- coding: utf-8 -*-
"""
方法一：ASAP（论文版）——SRP 取峰值。

Stage 1：方位谱 S(φ) = max_k P(φ, θ_k)（在 K 个粗俯仰角上取最大，把俯仰维"压掉"），
         先在粗方位网格上找峰，再在峰附近用细步长扫 P(φ, θ̂) 并做抛物线插值，锁定 φ̂。
Stage 2：固定 φ̂，在 θ̂ ± r 内先粗扫、再细扫俯仰角，最后用最小二乘抛物线拟合取顶点；
         拟合失败（不是开口向下，或顶点跑出窗口）时退回三点抛物线插值。

同时提供：
  - dense=True：Stage 1 方位用细步长整圈扫描（论文中的 "dense az. spectrum"）；
  - full_grid()：全网格 SRP-PHAT 作为参考（2° × 2°，共 180 × 46 = 8280 个方向，与论文一致）。

直接运行本文件会在几个随机方向上做一个小演示。
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from common import SRPFeatures, parabolic_vertex, wrap180


# ----------------------------------------------------------------------
# Stage 1：方位谱上的粗到细搜索（方法一、方法二共用）
# ----------------------------------------------------------------------
def stage1_azimuth(feat: SRPFeatures,
                   coarse_el: Sequence[float] = (0, 15, 30, 45, 60, 75),
                   az_coarse: float = 15.0, az_win: float = 15.0, az_fine: float = 1.0,
                   dense: bool = False) -> Tuple[float, float, Dict]:
    """返回 (φ̂, 粗俯仰 θ̂, 诊断信息)。"""
    coarse_el = np.asarray(coarse_el, dtype=float)
    step = az_fine if dense else az_coarse
    az_grid = np.arange(-180.0, 180.0, step)
    A, E = np.meshgrid(az_grid, coarse_el)                  # (K, Naz)
    P = feat.score_azel(A, E)
    S = P.max(axis=0)                                       # 方位谱
    k_best = P.argmax(axis=0)
    j = int(np.argmax(S))
    th_hat = float(coarse_el[k_best[j]])

    if dense:
        # 方位谱是一圈，取相邻点时要绕回
        off = parabolic_vertex(S[j - 1], S[j], S[(j + 1) % len(S)])
        phi_hat = float(wrap180(az_grid[j] + off * step))
    else:
        phi0 = az_grid[j]
        az_f = phi0 + np.arange(-az_win, az_win + 1e-9, az_fine)
        Pf = feat.score_azel(az_f, th_hat)
        i = int(np.argmax(Pf))
        off = parabolic_vertex(Pf[i - 1], Pf[i], Pf[i + 1]) if 0 < i < len(Pf) - 1 else 0.0
        phi_hat = float(wrap180(az_f[i] + off * az_fine))
    return phi_hat, th_hat, {"azimuth_spectrum": S, "az_grid": az_grid}


# ----------------------------------------------------------------------
# Stage 2：固定方位后的一维俯仰角细化（取峰值）
# ----------------------------------------------------------------------
def stage2_elevation_peak(feat: SRPFeatures, phi_hat: float, th_hat: float,
                          el_win: float = 20.0, el_coarse: float = 5.0, el_fine: float = 1.0,
                          ls_half: int = 3) -> float:
    lo, hi = max(0.0, th_hat - el_win), min(90.0, th_hat + el_win)
    tc_grid = np.arange(lo, hi + 1e-9, el_coarse)
    Pc = feat.score_azel(phi_hat, tc_grid)
    tc = tc_grid[int(np.argmax(Pc))]

    tf_grid = np.arange(max(lo, tc - el_coarse), min(hi, tc + el_coarse) + 1e-9, el_fine)
    Pf = feat.score_azel(phi_hat, tf_grid)
    i = int(np.argmax(Pf))

    # 多点最小二乘抛物线拟合
    a, b = max(0, i - ls_half), min(len(Pf), i + ls_half + 1)
    th = tf_grid[i]
    if b - a >= 3:
        c2, c1, _ = np.polyfit(tf_grid[a:b], Pf[a:b], 2)
        if c2 < 0:
            v = -c1 / (2.0 * c2)
            if tf_grid[0] <= v <= tf_grid[-1]:
                return float(np.clip(v, 0.0, 90.0))
    # 退回三点插值
    if 0 < i < len(Pf) - 1:
        th = tf_grid[i] + parabolic_vertex(Pf[i - 1], Pf[i], Pf[i + 1]) * el_fine
    return float(np.clip(th, 0.0, 90.0))


def asap_peak(feat: SRPFeatures, dense: bool = False, **kw) -> Tuple[float, float, Dict]:
    """ASAP：返回 (方位角, 俯仰角, 诊断信息)。"""
    feat.n_eval = 0
    s1 = {k: kw[k] for k in ("coarse_el", "az_coarse", "az_win", "az_fine") if k in kw}
    s2 = {k: kw[k] for k in ("el_win", "el_coarse", "el_fine", "ls_half") if k in kw}
    phi_hat, th_hat, _ = stage1_azimuth(feat, dense=dense, **s1)
    n1 = feat.n_eval
    th = stage2_elevation_peak(feat, phi_hat, th_hat, **s2)
    return phi_hat, th, {"n_eval": feat.n_eval, "n_eval_stage1": n1, "stage1_el": th_hat}


# ----------------------------------------------------------------------
# 参考：全网格 SRP-PHAT
# ----------------------------------------------------------------------
def full_grid(feat: SRPFeatures, az_step: float = 2.0, el_step: float = 2.0) -> Tuple[float, float, Dict]:
    feat.n_eval = 0
    az = np.arange(-180.0, 180.0, az_step)
    el = np.arange(0.0, 90.0 + 1e-9, el_step)
    A, E = np.meshgrid(az, el)
    P = feat.score_azel(A, E)                      # (Nel, Naz)
    ie, ia = np.unravel_index(int(np.argmax(P)), P.shape)
    da = parabolic_vertex(P[ie, ia - 1], P[ie, ia], P[ie, (ia + 1) % len(az)])
    de = parabolic_vertex(P[ie - 1, ia], P[ie, ia], P[ie + 1, ia]) if 0 < ie < len(el) - 1 else 0.0
    return (float(wrap180(az[ia] + da * az_step)),
            float(np.clip(el[ie] + de * el_step, 0.0, 90.0)),
            {"n_eval": feat.n_eval})


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from common import make_uca, make_source, simulate_freefield, az_error

    rng = np.random.default_rng(0)
    mics = make_uca()
    print(f"{'真值 az/el':>16} | {'全网格':>16} | {'ASAP-dense':>16} | {'ASAP-c2f':>16} | 打分次数(c2f)")
    for _ in range(6):
        az, el = rng.uniform(-180, 180), rng.uniform(0, 85)
        x = simulate_freefield(az, el, mics, make_source(rng=rng), rng=rng)
        f = SRPFeatures.from_signals(x, mics)
        g = full_grid(f)
        d = asap_peak(f, dense=True)
        c = asap_peak(f)
        print(f"{az:7.1f} {el:6.1f}   | {g[0]:7.1f} {g[1]:6.1f}   | {d[0]:7.1f} {d[1]:6.1f}   | "
              f"{c[0]:7.1f} {c[1]:6.1f}   | {c[2]['n_eval']}")
