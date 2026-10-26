# -*- coding: utf-8 -*-
"""
三种方法的对比实验（所有方法用同一段信号、同一份 SRP 特征）。

实验：
  A. 自由场 SNR 扫描（随机离网格方向，俯仰角 0–89° 均匀）
  B. 时延实现消融：整数截断（原代码）vs 分数时延
  C. 混响房间（pyroomacoustics，RT60 = 0.3 s / 0.6 s）

方法：
  full_grid       全网格 SRP-PHAT（2°×2°，8280 个方向，参考）
  asap_dense      ASAP，方位谱用 1° 整圈扫描
  asap_c2f        ASAP，方位谱粗到细（论文主方法）
  srp_mmse_flat   ASAP Stage 1 + SRP 后验均值（俯仰角均匀先验）
  srp_mmse_sph    同上，球面均匀先验 p(θ) ∝ cosθ
  tdoa_ml         GCC-PHAT 时延 + 加权最小二乘，直接反解
  tdoa_mmse       同上 + 高斯后验均值（俯仰角均匀先验）

用法：
  python run_compare.py            # 完整实验（几分钟）
  python run_compare.py --quick    # 快速试跑
输出在 results/ 目录：逐次结果 CSV、汇总表、图。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (SRPFeatures, angular_error, az_error, crb_std_deg, make_source, make_uca, noise_std_from_snr,
                    sample_directions, simulate_freefield, simulate_room, snr_from_noise_std, summarize)
from method1_asap_peak import asap_peak, full_grid
from method2_srp_mmse import calibrate_beta_by_peak, srp_mmse
from baseline_cfrc import cfrc
from method3_tdoa_bayes import tdoa_bayes

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_NOISE_STD = 0.6  # 与原代码一致：noise_sigma(0.5) * 1.2


def run_methods(feat: SRPFeatures, beta: float) -> Dict[str, Dict]:
    """在同一份特征上跑所有方法，返回 {方法名: {az, el, n_eval, time_ms}}。"""
    out = {}

    def timed(name: str, fn: Callable):
        t0 = time.perf_counter()
        az, el, info = fn()
        out[name] = {"az": az, "el": el, "n_eval": info.get("n_eval", 0),
                     "time_ms": 1e3 * (time.perf_counter() - t0)}

    timed("full_grid", lambda: full_grid(feat))
    timed("cfrc", lambda: cfrc(feat))
    timed("asap_dense", lambda: asap_peak(feat, dense=True))
    timed("asap_c2f", lambda: asap_peak(feat))
    timed("srp_mmse_flat", lambda: srp_mmse(feat, beta, prior="flat"))
    timed("srp_mmse_sph", lambda: srp_mmse(feat, beta, prior="sphere"))

    t0 = time.perf_counter()
    r = tdoa_bayes(feat, prior="flat")
    dt = 1e3 * (time.perf_counter() - t0)
    out["tdoa_ml"] = {"az": r["ml"][0], "el": r["ml"][1], "n_eval": 0, "time_ms": dt}
    out["tdoa_mmse"] = {"az": r["mmse"][0], "el": r["mmse"][1], "n_eval": 0, "time_ms": dt}
    return out


METHODS = ["full_grid", "cfrc", "asap_dense", "asap_c2f", "srp_mmse_flat", "srp_mmse_sph", "tdoa_ml", "tdoa_mmse"]
LABELS = {"full_grid": "全网格 SRP", "cfrc": "CFRC", "asap_dense": "ASAP-dense", "asap_c2f": "ASAP-c2f（取峰）",
          "srp_mmse_flat": "SRP 后验均值（均匀先验）", "srp_mmse_sph": "SRP 后验均值（球面先验）",
          "tdoa_ml": "TDOA-ML", "tdoa_mmse": "TDOA 后验均值"}


def records_for_trial(scenario: str, snr_db: float, az: float, el: float, res: Dict, crb, feat_ms: float,
                      **extra) -> List[Dict]:
    rows = []
    for m, r in res.items():
        rows.append({"scenario": scenario, "snr_db": snr_db, "true_az": az, "true_el": el, "method": m,
                     "est_az": r["az"], "est_el": r["el"], "az_err": float(az_error(r["az"], az)),
                     "el_err": abs(r["el"] - el),
                     "ang_err": float(angular_error(r["az"], r["el"], az, el)), "n_eval": r["n_eval"], "time_ms": r["time_ms"],
                     "feat_ms": feat_ms, "crb_az": crb[0], "crb_el": crb[1], **extra})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_test", type=int, default=150)
    ap.add_argument("--n_cal", type=int, default=60)
    ap.add_argument("--n_room", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.quick:
        args.n_test, args.n_cal, args.n_room = 30, 20, 10
    os.makedirs(args.out, exist_ok=True)

    mics = make_uca()
    rng_cal = np.random.default_rng(args.seed + 1)   # 校准和测试用不同的随机种子
    rng = np.random.default_rng(args.seed)
    ref_src = make_source(rng=np.random.default_rng(0))
    base_snr = snr_from_noise_std(ref_src, BASE_NOISE_STD)
    print(f"原代码的噪声设置对应单通道 SNR ≈ {base_snr:.2f} dB")

    # ---------------- β 校准（单独的数据，覆盖多个 SNR；按 SRP 峰值高度自适应）----------------
    cal_feats, cal_el = [], []
    cal_snrs = [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
    for az, el in sample_directions(args.n_cal * len(cal_snrs) // 2, rng_cal):
        s = make_source(rng=rng_cal)
        ns = noise_std_from_snr(s, float(rng_cal.choice(cal_snrs)))
        x = simulate_freefield(az, el, mics, s, noise_std=ns, rng=rng_cal)
        cal_feats.append(SRPFeatures.from_signals(x, mics))
        cal_el.append(el)
    beta, beta_info = calibrate_beta_by_peak(cal_feats, cal_el, n_bins=3)
    beta_table = {"edges": beta_info["edges"], "betas": beta.betas}
    print(f"β 校准结果：{beta}")

    rows: List[Dict] = []

    # ---------------- A. 自由场 SNR 扫描 ----------------
    snr_list = [-10.0, -5.0, 0.0, round(base_snr, 2), 5.0, 10.0, 20.0]
    for snr in snr_list:
        t0 = time.time()
        for az, el in sample_directions(args.n_test, rng):
            s = make_source(rng=rng)
            ns = noise_std_from_snr(s, snr)
            x = simulate_freefield(az, el, mics, s, noise_std=ns, rng=rng)
            tf = time.perf_counter()
            feat = SRPFeatures.from_signals(x, mics)
            feat_ms = 1e3 * (time.perf_counter() - tf)
            res = run_methods(feat, beta)
            rows += records_for_trial("freefield", snr, az, el, res, crb_std_deg(az, el, mics, s, ns), feat_ms)
        print(f"[A] SNR = {snr:6.2f} dB 完成，用时 {time.time() - t0:.1f}s")

    # ---------------- B. 时延实现消融（基准 SNR）----------------
    for mode in ("integer", "fractional"):
        for az, el in sample_directions(args.n_test, rng):
            s = make_source(rng=rng)
            x = simulate_freefield(az, el, mics, s, noise_std=BASE_NOISE_STD, delay_mode=mode, rng=rng)
            feat = SRPFeatures.from_signals(x, mics)
            res = {k: v for k, v in run_methods(feat, beta).items() if k in ("full_grid", "cfrc", "asap_c2f")}
            rows += records_for_trial(f"delay_{mode}", base_snr, az, el, res,
                                      crb_std_deg(az, el, mics, s, BASE_NOISE_STD), 0.0)
    print("[B] 时延消融完成")

    # ---------------- C. 混响房间 ----------------
    for rt60 in (0.3, 0.6):
        t0 = time.time()
        n_done = 0
        while n_done < args.n_room:
            az, el = sample_directions(1, rng, el_max=70.0)[0]
            s = make_source(rng=rng)
            ns = noise_std_from_snr(s, 10.0) * 0.5  # 直达声在 2 m 处有衰减，这里噪声取得小一些，混响为主
            x = simulate_room(az, el, mics, s, rt60=rt60, noise_std=ns, rng=rng)
            if x is None:
                continue
            feat = SRPFeatures.from_signals(x, mics)
            res = run_methods(feat, beta)
            rows += records_for_trial(f"room_rt{rt60}", float("nan"), az, el, res, (np.nan, np.nan), 0.0)
            n_done += 1
        print(f"[C] RT60 = {rt60}s 完成，用时 {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "all_trials.csv"), index=False)

    # ---------------- 汇总表 ----------------
    summ = []
    for (sc, snr, m), g in df.groupby(["scenario", "snr_db", "method"], dropna=False, sort=False):
        s = summarize(g.az_err.values, g.el_err.values, g.true_el.values, ang_err=g.ang_err.values)
        s.update({"scenario": sc, "snr_db": snr, "method": m, "n_eval_mean": g.n_eval.mean(),
                  "time_ms_mean": g.time_ms.mean(),
                  "crb_el_median": float(np.nanmedian(g.crb_el)) if g.crb_el.notna().any() else np.nan,
                  "crb_az_median": float(np.nanmedian(g.crb_az[g.true_el < 85]))
                  if g.crb_az.notna().any() else np.nan})
        summ.append(s)
    S = pd.DataFrame(summ)
    cols = ["scenario", "snr_db", "method", "ang_rmse", "ang_median", "ang_outlier_rate", "az_rmse", "az_median", "az_outlier_rate", "el_rmse",
            "el_median", "el_outlier_rate", "crb_az_median", "crb_el_median", "n_eval_mean", "time_ms_mean", "n"]
    S = S[cols]
    S.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(S.round(3).to_markdown(index=False))
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"beta": repr(beta), "beta_table": beta_table, "base_snr_db": base_snr,
                   "feat_ms_mean": float(df.loc[df.feat_ms > 0, "feat_ms"].mean())}, f, ensure_ascii=False, indent=2)

    make_figures(df, base_snr, args.out)
    print("完成，结果在", args.out)


def make_figures(df: pd.DataFrame, base_snr: float, out: str):
    styles = {"full_grid": ("k", "-", "o"), "cfrc": ("tab:orange", "-", "x"), "asap_dense": ("tab:gray", "--", "s"),
              "asap_c2f": ("tab:blue", "-", "s"), "srp_mmse_flat": ("tab:green", "-", "^"),
              "srp_mmse_sph": ("tab:olive", "--", "^"), "tdoa_ml": ("tab:red", "-", "v"),
              "tdoa_mmse": ("tab:purple", "-", "D")}
    bins = np.arange(0, 91, 10)
    centers = 0.5 * (bins[1:] + bins[:-1])

    # 图 1：基准 SNR 下，误差随真实俯仰角的变化
    d = df[(df.scenario == "freefield") & (np.isclose(df.snr_db, round(base_snr, 2)))]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, err, crb, title in ((axs[0], "el_err", "crb_el", "俯仰角 RMSE"), (axs[1], "az_err", "crb_az", "方位角 RMSE")):
        for m in METHODS:
            g = d[d.method == m]
            if err == "az_err":
                g = g[g.true_el < 85]
            y = [np.sqrt(np.mean(g[(g.true_el >= a) & (g.true_el < b)][err] ** 2)) for a, b in zip(bins[:-1], bins[1:])]
            c, ls, mk = styles[m]
            ax.plot(centers, y, color=c, ls=ls, marker=mk, ms=4, label=LABELS[m])
        g = d[d.method == "full_grid"]
        if err == "az_err":
            g = g[g.true_el < 85]
        yc = [np.median(g[(g.true_el >= a) & (g.true_el < b)][crb]) for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(centers, yc, color="k", ls=":", lw=2, label="CRB（区间中位数）")
        ax.set_yscale("log")
        ax.set_xlabel("真实俯仰角（°）")
        ax.set_ylabel("RMSE（°，对数坐标）")
        ax.set_title(f"{title}（自由场，SNR = {base_snr:.1f} dB）")
        ax.grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_rmse_vs_elevation.png"), dpi=200)
    plt.close(fig)

    # 图 2：RMSE 随 SNR 变化
    d = df[df.scenario == "freefield"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, err, crb, title in ((axs[0], "el_err", "crb_el", "俯仰角 RMSE"), (axs[1], "az_err", "crb_az", "方位角 RMSE")):
        snrs = sorted(d.snr_db.unique())
        for m in METHODS:
            g = d[d.method == m]
            if err == "az_err":
                g = g[g.true_el < 85]
            y = [np.sqrt(np.mean(g[np.isclose(g.snr_db, s)][err] ** 2)) for s in snrs]
            c, ls, mk = styles[m]
            ax.plot(snrs, y, color=c, ls=ls, marker=mk, ms=4, label=LABELS[m])
        g = d[d.method == "full_grid"]
        if err == "az_err":
            g = g[g.true_el < 85]
        # θ→0° 时俯仰 CRB 趋于无穷，用中位数比 RMS 稳定
        ax.plot(snrs, [np.median(g[np.isclose(g.snr_db, s)][crb]) for s in snrs],
                color="k", ls=":", lw=2, label="CRB（中位数）")
        ax.set_yscale("log")
        ax.set_xlabel("单通道 SNR（dB）")
        ax.set_ylabel("RMSE（°，对数坐标）")
        ax.set_title(title + "（自由场，俯仰角 0–89° 均匀）")
        ax.grid(alpha=0.3, which="both")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_rmse_vs_snr.png"), dpi=200)
    plt.close(fig)

    # 图 3：混响房间
    d = df[df.scenario.str.startswith("room")]
    if len(d):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
        scen = sorted(d.scenario.unique())
        w = 0.8 / len(METHODS)
        for ax, err, title in ((axs[0], "el_err", "俯仰角 RMSE"), (axs[1], "az_err", "方位角 RMSE")):
            for i, m in enumerate(METHODS):
                y = [np.sqrt(np.mean(d[(d.scenario == s) & (d.method == m)][err] ** 2)) for s in scen]
                ax.bar(np.arange(len(scen)) + (i - len(METHODS) / 2 + 0.5) * w, y, w,
                       color=styles[m][0], label=LABELS[m])
            ax.set_xticks(np.arange(len(scen)))
            ax.set_xticklabels([s.replace("room_rt", "RT60 = ") + " s" for s in scen])
            ax.set_ylabel("RMSE（°）")
            ax.set_title(title + "（混响房间，声源距离 2 m）")
            ax.grid(alpha=0.3, axis="y")
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig3_reverb.png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
