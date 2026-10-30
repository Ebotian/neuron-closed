# 架构：三层 + 一条规则

## 为什么要分

一次会话里，同一个 bug（对 `|slot|` 除两次）**犯了两次**，同一份分析（split-half）**写了两份**，device 不匹配**遇到五次**，
Lean 文件因为整文件切片**改坏三次**。它们不是五个错误，是同一个结构性原因：**同一个泛函在多处各写一遍**。

## 三层

```
core/           无实验知识，只管"物理量怎么换算、张量放哪、模型怎么读、闭式解怎么解"
  units.py        p50/‖W_col‖：剂量单位的唯一 owner（曾经散在 13 个脚本里）
  design.py       device/dtype 策略；scale_for 只发同设备张量；check() 在边界拒错
  sampling.py     采样协议（种子顺序固定，否则轨迹变、下游全变）
  loading.py      冻结模型的加载（device_map 直落 GPU；与 vendor 逐值一致）
  cache.py        p50/列范数按冻结权重缓存；strict=False 生产用，校验在 verify_cache.py
  closedform.py   预算约束下的最优解（M 满支撑奇异 ⇒ 最小范数 M⁺，薄 SVD）
estimators/     一个泛函一个文件
  phi.py          路径版 J2（nat），主构念
  judge.py        判官版 φ（plan 2.40 的构念；与上面并列存在，差异本身是结论）
  score.py        逐 rollout、未加权、整序列密度；每上下文一次 batched backward
  pathwise.py     沿轨迹的 teacher-forced 梯度；2K 对一个批，分块
  teacher.py      固定前缀的 teacher-forced 梯度（均值，不是求和）
  steer.py        注入与读数（向量化，每层一次加法）
analysis/       只吃 Gram/数值，不吃模型
  split_half.py   唯一实现（Gram 版）；暴力参照验证
  icc.py          纯 Python：输入 (Gram, group_sizes, k)，输出 ICC + F 区间
  report.py       汇总 dict 的唯一产地
  quotient.py     残差空间映射 to_v、全层直和余弦 cos_direct、匹配零分布 null_for、report_list/q3_verdict
  fd.py           有限差分判据本身（G1 逐坐标 min-over-eps < 2%，G2 旧法必须 > 5% 可见）
scripts/        现役薄壳（全部 ≤120 行）：组装 + 冻结判据 + 写 results
  p_a2_pathj2.py 119      p_a_reinforce.py 98     noise_scale.py 74
  p_a4_closed_form.py 91  fd_gate.py 120          quotient_compare_v2.py 74
  reliability.py 21（弃用 shim）                  legacy_capture.py 42（唯一的阴性对照）
  worker.py 79    常驻模型，按 stdin 每行跑一次 pass（见下文）
  bench_8x32.py 99  两次 pass + PASS_START/PASS_END 打点
  verify_*.py     逐值验证：units/design/loading/estimators/pathwise/split_half/icc/cache
```

## 一条规则

> **一个泛函，一份实现。** 需要不同的归一化/度量，就显式传参数，不许在调用处再写一遍公式。

由 `make check` 的 `noregress` 项对**现役链**机械执行，四条规则：

1. `p50/‖W_col‖` 只许出现在 `core/units.py`（曾散在 13 个脚本里，同一个泛函长出三种归一化）；
2. `detach().clone().requires_grad_` 只许出现在 `legacy_capture.py`（唯一的阴性对照；它曾悄悄切断残差载波路径）；
3. `split_half` 只许有一处实现（两份拷贝造出过单边 Gram 加权）；
4. **层模块拥有的函数不许在壳里重定义**——`quotient_compare_v2.py` 就曾在 import 旁边又写了一遍
   `def to_v/cos_direct/null_for`，把 import 遮蔽掉、其中两份成了死代码。这条数值上完全无害，所以逐值回归看不见它，
   只有 grep 闸门能抓。规则 4 是审计第 1 条的直接产物。

四条都是用**故意注入违规**验证过的：注入 → 闸门红；恢复 → 绿。

## 验证层级（便宜的在前）

| 层 | 怎么做 | 需要 GPU? |
|---|---|---|
| 单位/设备 | `verify_units.py`、`verify_design.py`：逐值 `torch.equal` | 是（要模型） |
| 仪器 | `fd_gate.py`：fp32 有限差分对**出货模块** | 是 |
| 估计量 | `verify_pathwise.py`、`verify_split_half.py`：对照**逐字复制**的旧实现或**暴力枚举** | pathwise 要，split-half 不要 |
| 分析 | `verify_icc.py`：从存下的 Gram 重算，纯 Python | **不要** |
| 加载 | `verify_loading.py`：与 vendor 加载器逐值一致 | 是 |
| 记录 | `make check` 的 `instruments` 项：Lean sha + 校准门绑定仪器哈希 | 不要 |

**关键一条**：重构的参照是**紧邻的上一个 commit**，不是更早的存档记录。本次两次"失败"都是因为拿旧记录当参照，
而旧记录本身已不可复现（plan 250、plan 253）。

## make certify

`make certify` 重跑校准门与 Lean 审计并**重写** records；`make check` 的 `instruments` 项随后校验 sha 一致。
证书不再靠人记得更新——改了仪器却不重认证，`make check` 会红。

## 常驻 worker（`scripts/worker.py`）

每个脚本各自载入 8.5 GiB 权重，监控上表现为显存 0 → 28 GiB → 0 的锯齿。`worker.py` 只载入一次，之后按 stdin 每行跑一次
8×32 pass，用 `PASS_START`/`PASS_END` 打点给外部采样器对齐。任务定义因此是"常驻模型上的一次 pass"：

| 场景 | 墙钟 | GPU 均值 |
|---|---|---|
| warm 任务 1 | 33.0 s | 72.5 % |
| warm 任务 2 | 29.5 s | 86.6 % |
| 冷启动单次（含装载） | 43.6 s | 56.3 % |

装载是一次性会话成本，单独报告，不塞进任务预算，也不藏起来。数值与单发运行一致（`MS_w 39.536593`、`ICC 0.00345447`、
`|pw| 134.1485`）。
