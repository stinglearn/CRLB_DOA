# -*- coding: utf-8 -*-
"""
精度 – 计算量（Pareto）曲线：ASAP 和 CFRC 各扫一组参数，在同样的信号上比较。

论文里只给一个工作点（206 次 vs 155 次）说服力不够；这张图能回答
"在同样的打分次数下，谁更准"。每个点是一组参数，横轴是平均打分次数，纵轴是球面夹角 RMSE。

用法：
  python pareto.py                    # 默认：SNR 1.5 dB 和 -5 dB，各 200 个随机方向
  python pareto.py --n 500 --snr 1.5 0 -5
输出：results_pareto/pareto.csv、pareto.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (SRPFeatures, angular_error, make_source, make_uca, noise_std_from_snr,
                    sample_directions, simulate_freefield)
from method1_asap_peak import asap_peak, full_grid
from baseline_cfrc import cfrc

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EL4 = (0, 30, 60, 85)
EL5 = (0, 20, 40, 60, 85)
EL7 = (0, 15, 30, 45, 60, 75, 85)

ASAP_CONFIGS = [
    dict(az_coarse=30, coarse_el=EL4, az_win=30, az_fine=2, el_fine=2, n_cand=1),
    dict(az_coarse=30, coarse_el=EL4, az_win=30, az_fine=2, el_fine=1, n_cand=2),
    dict(az_coarse=20, coarse_el=EL5, az_win=20, az_fine=2, el_fine=1, n_cand=1),
    dict(az_coarse=20, coarse_el=EL5, az_win=20, az_fine=1, el_fine=1, n_cand=2),
    dict(az_coarse=15, coarse_el=EL7, az_win=15, az_fine=1, el_fine=1, n_cand=1),
    dict(az_coarse=15, coarse_el=EL7, az_win=15, az_fine=1, el_fine=1, n_cand=2),   # 默认
    dict(az_coarse=10, coarse_el=EL7, az_win=10, az_fine=1, el_fine=1, n_cand=2),
    dict(az_coarse=10, coarse_el=EL7, az_win=10, az_fine=0.5, el_fine=0.5, n_cand=3),
]

CFRC_CONFIGS = [
    dict(n_az0=8, n_el0=4, n_az=5, n_el=5, top_n=3, n_iter=3),
    dict(n_az0=12, n_el0=5, n_az=5, n_el=5, top_n=3, n_iter=4),
    dict(n_az0=12, n_el0=5, n_az=6, n_el=6, top_n=3, n_iter=4),
    dict(n_az0=12, n_el0=5, n_az=6, n_el=6, top_n=3, n_iter=6),   # 默认
    dict(n_az0=12, n_el0=5, n_az=6, n_el=6, top_n=2, n_iter=7),
    dict(n_az0=16, n_el0=6, n_az=7, n_el=7, top_n=3, n_iter=6),
    dict(n_az0=24, n_el0=7, n_az=8, n_el=8, top_n=4, n_iter=7),
    dict(n_az0=24, n_el0=7, n_az=10, n_el=10, top_n=4, n_iter=8),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--snr", type=float, nargs="+", default=[1.5, -5.0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results_pareto")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mics = make_uca()
    rows = []
    for snr in args.snr:
        rng = np.random.default_rng(args.seed)
        feats, truth = [], []
        for az, el in sample_directions(args.n, rng):
            s = make_source(rng=rng)
            x = simulate_freefield(az, el, mics, s, noise_std=noise_std_from_snr(s, snr), rng=rng)
            feats.append(SRPFeatures.from_signals(x, mics))
            truth.append((az, el))

        def run(name, cid, fn, cfg):
            err, nev = [], []
            for f, (az, el) in zip(feats, truth):
                a, e, d = fn(f)
                err.append(float(angular_error(a, e, az, el)))
                nev.append(d["n_eval"])
            err = np.array(err)
            rows.append({"snr_db": snr, "method": name, "config_id": cid, "config": str(cfg),
                         "n_eval": float(np.mean(nev)), "ang_rmse": float(np.sqrt(np.mean(err ** 2))),
                         "ang_median": float(np.median(err)), "outlier_rate": float(np.mean(err > 10))})

        for i, cfg in enumerate(ASAP_CONFIGS):
            run("ASAP", i, lambda f, c=cfg: asap_peak(f, **c), cfg)
        for i, cfg in enumerate(CFRC_CONFIGS):
            run("CFRC", i, lambda f, c=cfg: cfrc(f, **c), cfg)
        run("全网格", 0, full_grid, {})
        print(f"SNR = {snr} dB 完成")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "pareto.csv"), index=False)

    fig, axs = plt.subplots(1, len(args.snr), figsize=(6 * len(args.snr), 4.3), squeeze=False)
    for ax, snr in zip(axs[0], args.snr):
        d = df[df.snr_db == snr]
        for name, c, mk in (("ASAP", "tab:blue", "s"), ("CFRC", "tab:orange", "x")):
            g = d[d.method == name].sort_values("n_eval")
            ax.plot(g.n_eval, g.ang_rmse, color=c, marker=mk, label=f"{name}（RMSE）")
            ax.plot(g.n_eval, g.ang_median, color=c, marker=mk, ls=":", alpha=0.7, label=f"{name}（中位数）")
        g = d[d.method == "全网格"]
        ax.axhline(float(g.ang_rmse.iloc[0]), color="k", ls="--", lw=1, label="全网格 RMSE（8280 次）")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("平均打分次数（对数坐标）")
        ax.set_ylabel("球面夹角误差（°，对数坐标）")
        ax.set_title(f"精度 – 计算量（自由场，SNR = {snr} dB）")
        ax.grid(alpha=0.3, which="both")
    axs[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "pareto.png"), dpi=200)
    print("完成，结果在", args.out)


if __name__ == "__main__":
    main()
