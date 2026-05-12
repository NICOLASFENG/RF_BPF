# RF BPF Capacitor Optimization Code

这个仓库保存的是一个用于 TGV-IPD 可重构射频带通滤波器的 C++ 优化程序。代码目标不是直接生成版图 PCell，而是在理想 LC 和有限约束下搜索 C1-C9 的目标电容值，检查 N77、N78、N79 工作模式的通带、阻带、谐波抑制和传输零点位置，为后续 schematic、PDK 器件模型和 PEX 校正提供初始数值。

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `finalcheck(a1).cpp` | 对应论文中的 Algorithm 1。它是约束收敛后的最终检查程序，当前默认只打开 N78，使用紧凑 GA 对单个目标 band 做最终 re-optimization / final check。 |
| `firstcheck(a2).cpp` | 对应论文中 Algorithm 2-7 的完整探索框架。它同时考虑 N77/N78/N79 三个模式，支持 RF-only 搜索、共享压缩、硬件代价、baseline 偏移约束、GA/DE/jDE/hybrid 等策略。 |
| `branch.cpp` | 可用于版图参数定位，帮助初步估算开关电容分支相关参数。这个计算略为粗糙，如果是有 RF 设计经验的人可以不用。 |
| `tools/find_feasible.cpp` | 基于 `firstcheck(a2).cpp` 的辅助可行解搜索器。它把主函数临时重命名后复用内部 RF 评估函数，按单个 mode 寻找更容易满足 dense check 的候选行。 |
| `tools/run_active_gpp.ps1` | Windows 下编译并运行当前 C++ 源文件的脚本，输出到本地 `build` 文件夹。 |
| `tools/start_active_gpp_run.ps1` | 启动上面 runner 的 PowerShell 包装脚本。 |

`build` 目录只放本地编译出的 exe、log 和临时结果，不属于源码逻辑。

## 物理和电路模型

代码把滤波器看成一个 2-port LC 网络。两个端口节点为 0 和 8，中间节点为 1-7。电容 C1-C9 和固定电感 L1-L5 通过节点导纳矩阵描述：

- 电容支路使用 `j*w*C` 写入导纳矩阵。
- 电感支路使用 `1/(j*w*L)` 写入导纳矩阵。
- 内部 7 个节点通过高斯消元消去，得到端口等效导纳矩阵 `Yeff`。
- 再用 50 ohm 参考阻抗把 `Yeff` 转为 S 参数，得到 `S11` 和 `S21`。

核心函数是：

- `calculate_s_parameters(...)`：建立 MNA 导纳矩阵并计算 S 参数。
- `evaluate_mode(...)` / `evaluate_candidate(...)`：把 S 参数扫描结果转换成 return loss、insertion loss、阻带 rejection、谐波 rejection 和 penalty。
- `compute_transmission_zeros(...)`：根据 LC 组合估计传输零点。

传输零点检查主要使用以下关系：

- `TZ1 = 1 / (2*pi*sqrt((C3+C4)*L3))`
- `TZ2 = 1 / (2*pi*sqrt((C6+C7)*L4))`
- `TZ3 = 1 / (2*pi*sqrt(C9*L2))`
- `TZ4 = 1 / (2*pi*sqrt(C1*L1))`

这些值用于约束低侧零点、高侧零点和通带边缘之间的相对位置。

## 评价指标

每个候选电容表都会在多个频率网格上计算：

- 通带 `S11` return loss 是否达到目标值。
- 通带 `S21` insertion loss 是否低于目标值。
- lower stopband 和 upper stopband 的最小 rejection。
- 二次、三次谐波附近的最小 rejection。
- 传输零点是否落在合理窗口。
- 三个模式之间电容能否共享，以及共享后的硬件复杂度。

`firstcheck(a2).cpp` 采用 coarse、mid、dense 三层频率网格。coarse 用于快速排序，mid/dense 用于周期性验证强候选，减少只在稀疏采样点上表现好的假可行解。

## firstcheck(a2) 的算法流程

`firstcheck(a2).cpp` 是完整探索器，主要流程如下：

1. 定义 N77/N78/N79 band、stopband、harmonic window 和基准电容表。
2. 对 baseline 做 repair：边界裁剪、0.01 pF 量化、可选共享 mask 强制。
3. 初始化 population：baseline seed、局部 jitter、宽范围 jitter、随机采样、seed-random blend、共享 pattern seed。
4. 用 coarse grid 评估全部候选。
5. 每隔若干代对 top candidates 做 mid/dense grid 复查。
6. 根据配置选择 GA、DE、jDE 或 hybrid 更新种群。
7. 对可行或接近可行的候选维护 archive。
8. 可选执行 progressive sharing：逐步尝试 N77/N78、N77/N79、N78/N79 或 all-mode 共享，只接受 RF 可行性仍成立的压缩。
9. 输出最佳电容表、decomposition、metrics、failure diagnostics 和 lambda sweep CSV。

适应度函数分两层：

- 不可行时，主要按 RF penalty 和 worst violation 排序。
- 可行后，再加入 sharing cost、hardware cost 和 baseline distance，让结果既满足 RF，又更容易落地成少分支、少差异的开关电容网络。

常用编译命令：

```bash
g++ -std=c++17 -O3 -march=native -fopenmp "firstcheck(a2).cpp" -o ctc_ga
```

常用运行方式：

```bash
./ctc_ga --two-stage --progressive-sharing --local-polish --output-prefix ctc_ga
./ctc_ga --rf-only --output-prefix rf_only
./ctc_ga --eval-baseline-only
```

常见输出：

- `*_best_cap_table.csv`：最终三模式 C1-C9 目标表。
- `*_decomposition.csv`：共享/分支拆分结果。
- `*_metrics.csv`：每个模式的 RF 指标。
- `*_failure_diagnostics.csv`：未满足约束时的失败原因。
- `*_lambda_sweep.csv`：不同共享权重下的 trade-off。

## finalcheck(a1) 的算法流程

`finalcheck(a1).cpp` 是较紧凑的最终检查版本。它把一个目标 band 的自由电容编码成 GA 基因，默认固定 C1 和 L1-L5，搜索 C2-C9 的组合。当前 main 函数默认启用 N78，N77 和 N79 band 定义保留为注释模板，方便切换。

主要流程：

1. 根据目标 band 生成 passband、lower stopband、upper stopband、二次和三次谐波采样点。
2. 根据 band center 和传输零点经验约束生成搜索边界。
3. 用 seed bank 加随机候选建立初始种群。
4. 在 coarse grid 上快速评估并排序。
5. 每隔 `dense_check_interval` 在 dense grid 上检查当前最优候选。
6. 可行后继续 polish 若干代，直到停滞或达到代数上限。
7. 打印最终 C2-C9、电路固定值、各项 RF 条件和传输零点位置。

常用编译命令：

```bash
g++ -std=c++17 -O3 -march=native "finalcheck(a1).cpp" -o finalcheck
```

常用运行方式：

```bash
./finalcheck --population 360 --generations 900 --restarts 4
./finalcheck --seed 20260427 --polish-generations 160
```

## tools/find_feasible.cpp

这个工具用于单 mode 可行性搜索。它直接 include `firstcheck(a2).cpp`，复用其中的 band、bounds、baseline、S 参数计算和 dense evaluation，但使用更专门的 per-mode 搜索流程。

适合用在以下场景：

- 某一个 mode 的 dense 指标一直不通过，需要单独找更好的初始行。
- 想先得到 N77、N78 或 N79 的可行 row，再回填到完整三模式优化器。
- 想排查失败是来自某个 mode 的 RF 可行性，还是来自共享压缩。

编译示例：

```bash
g++ -std=c++17 -O3 -march=native -fopenmp "tools/find_feasible.cpp" -o find_feasible
```

运行示例：

```bash
./find_feasible --mode N78 --population 1600 --generations 1600 --threads 8
```

## 注意事项

本项目输出的是 ideal LC target capacitance。真正用于版图和 tape-out 前，还必须用 PDK 的 MIM 电容、NMOS 开关 Ron/Coff、有限 Q 电感、substrate loss、routing parasitic 和 post-layout extraction 重新校正。也就是说，这里的程序负责找到“电路拓扑和电容目标值的可行起点”，不是最终物理尺寸签核工具。

如果需要上传结果，请只提交源码和必要说明。`build`、exe、log、CSV 和论文手稿都应视为本地运行产物或写作资料。
