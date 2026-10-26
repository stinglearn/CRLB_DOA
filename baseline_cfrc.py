# -*- coding: utf-8 -*-
"""
【修改 4】对比基线：CFRC（coarse-to-fine region contraction，Do & Silverman, WASPAA 2007）。

思路：在整个搜索区域上放一张粗网格打分；保留得分最高的 top_n 个点；
把搜索区域收缩成能包住这些点的最小方框（再向外扩一个网格步长）；在新方框里重新铺网格；
重复 n_iter 轮。和 ASAP 不同，它对方位和俯仰一视同仁，在二维角度空间里一起收缩。

这里按原论文思路做了一个确定性的二维（方位 × 俯仰）实现，参数可调，便于画 Pareto 曲线。
原论文是在三维空间坐标里收缩，用于近场定位；这里只做方向，属于合理简化，论文里要说明。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from common import SRPFeatures, wrap180


def cfrc(feat: SRPFeatures, n_az0: int = 12, n_el0: int = 5, n_az: int = 6, n_el: int = 6,
         top_n: int = 3, n_iter: int = 6, min_span: float = 0.5) -> Tuple[float, float, Dict]:
    """返回 (方位角, 俯仰角, 诊断信息)。

    第 1 轮用 n_az0 × n_el0 的网格覆盖整个上半球，之后每轮用 n_az × n_el 的网格。
    默认参数下 60 + 5 × 36 = 240 次打分，和修改后的 ASAP（约 245 次）相当。
    这组默认值是在试跑中调出来的、精度较好的一组，避免把基线设得太弱（审稿人会质疑"稻草人"对比）。
    完整的公平比较请用 pareto.py 扫参数画曲线。
    """
    feat.n_eval = 0
    best_s, best_az, best_el = -np.inf, 0.0, 0.0
    # 第 1 轮：方位整圈，俯仰 [0, 90]
    az_c, az_span = 0.0, 360.0
    el_lo, el_hi = 0.0, 90.0
    for it in range(n_iter):
        na, ne = (n_az0, n_el0) if it == 0 else (n_az, n_el)
        if it == 0:
            az = np.arange(na) * (360.0 / na) - 180.0
        else:
            az = az_c + np.linspace(-az_span / 2.0, az_span / 2.0, na)
        el = np.linspace(el_lo, el_hi, ne)
        A, E = np.meshgrid(az, el)
        P = feat.score_azel(A, E).ravel()
        A, E = A.ravel(), E.ravel()
        k = int(np.argmax(P))
        if P[k] > best_s:
            best_s, best_az, best_el = float(P[k]), float(wrap180(A[k])), float(E[k])

        # 收缩：包住前 top_n 个点的方框（方位以当前最好点为中心展开，避免 ±180° 绕回问题）
        top = np.argsort(P)[::-1][:top_n]
        d_az = wrap180(A[top] - best_az)
        step_az = (az[1] - az[0]) if len(az) > 1 else az_span
        step_el = (el[1] - el[0]) if len(el) > 1 else (el_hi - el_lo)
        lo_a, hi_a = d_az.min() - step_az, d_az.max() + step_az
        az_c = float(wrap180(best_az + 0.5 * (lo_a + hi_a)))
        az_span = float(min(max(hi_a - lo_a, min_span), 360.0))
        el_lo = float(max(0.0, E[top].min() - step_el))
        el_hi = float(min(90.0, E[top].max() + step_el))
        if el_hi - el_lo < min_span:
            mid = 0.5 * (el_lo + el_hi)
            el_lo, el_hi = max(0.0, mid - min_span / 2), min(90.0, mid + min_span / 2)
    return best_az, best_el, {"n_eval": feat.n_eval}


if __name__ == "__main__":
    from common import make_uca, make_source, simulate_freefield, angular_error

    rng = np.random.default_rng(3)
    mics = make_uca()
    for _ in range(6):
        az, el = rng.uniform(-180, 180), rng.uniform(0, 89)
        f = SRPFeatures.from_signals(simulate_freefield(az, el, mics, make_source(rng=rng), rng=rng), mics)
        a, e, d = cfrc(f)
        print(f"真值 {az:7.1f} {el:5.1f} → CFRC {a:7.1f} {e:5.1f} | 夹角误差 {float(angular_error(a, e, az, el)):.2f}° | 打分 {d['n_eval']}")
