# 平面阵列 3D DOA：三种方法对比代码

在原来"条带 + 两点之间搜索"（CCFRC_BTP）代码的基础上重写，三种方法共用同一套仿真和 SRP-PHAT 特征，保证对比公平。

## 文件

| 文件 | 内容 |
|---|---|
| `common.py` | 阵列几何、信号仿真（自由场 / pyroomacoustics 混响）、SRP-PHAT 特征、CRB、评估指标 |
| `method1_asap_peak.py` | 方法一：ASAP（论文版），Stage 1 方位谱粗到细，Stage 2 俯仰一维细化取峰值；另含全网格 SRP 参考 |
| `method2_srp_mmse.py` | 方法二：ASAP Stage 1 + SRP 后验均值（MMSE），含 β 校准函数 |
| `method3_tdoa_bayes.py` | 方法三：GCC-PHAT 时延 + 加权最小二乘求水平分量 q，再做高斯后验（ML / MAP / MMSE） |
| `run_compare.py` | 对比实验：SNR 扫描、时延实现消融、混响房间；输出 CSV、汇总表和图 |

每个 `method*.py` 都可以单独运行，会打印几个随机方向上的小演示。

## 运行

```bash
pip install numpy pandas matplotlib tabulate pyroomacoustics
python method1_asap_peak.py        # 单独试方法一
python run_compare.py --quick      # 快速试跑（约 40 秒）
python run_compare.py              # 完整实验（几分钟）
```

结果在 `results/`：`all_trials.csv`（逐次结果）、`summary.csv` / `summary.md`（汇总表）、`fig1~3*.png`。

## 和原代码的对应关系与改动

保持一致：8 元均匀圆阵（直径 88.75 mm）、fs = 50 kHz、0.5 s chirp（500–2500 Hz）+ 共享噪声 0.1、
独立噪声 std = 0.6（对应单通道 SNR ≈ 1.5 dB）、Hamming 20 ms / hop 10 ms / n_fft 1024、
先对帧平均互谱再 PHAT 归一化的频域 SRP。

改动（都可以切换回原样）：

1. **分数时延**：原代码用 `.long()` 把时延截断成整数采样点（最大时延只有约 13 个采样点），
   会引入额外误差。默认改为频域相移的精确时延，`delay_mode="integer"` 可复现原做法。
2. **只算一次 SRP**：原代码把平均后的协方差 repeat 回每一帧再逐帧计算，同一个结果重复算了约 50 次。
3. **只用 i<j 的麦克风对**：原代码包含 i=j 的自相关项（只是常数偏移，不影响峰值位置）。
4. **两种打分后端，结果等价**：`"freq"` 逐频点求和（和原代码一致），`"lut"` 先 IFFT 成 GCC
   再查表（论文式 (2) 的形式）。两者相关系数 1.0000，但全网格 8280 个方向，`freq` 约 3.6 s，`lut` 约 7 ms。
5. **没有用 torch**，只依赖 NumPy，方便在任何机器上跑。

## 注意

- β（方法二的温度参数）在**单独的校准数据**上选，不能用测试数据选。
- 方法三的后验依赖"时延误差近似高斯"的假设；GCC 挑错峰时这个假设不成立。
- 后验均值保证的是在先验分布下的**平均**误差最小，不保证每个真实角度都更准，应按俯仰角分段报告误差。
