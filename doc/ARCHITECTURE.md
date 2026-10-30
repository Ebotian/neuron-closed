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
  loading.py      冻结模型的加载（low_cpu_mem_usage；与 vendor 逐值一致）
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
experiments/    四个薄壳（≤120 行）：组装 + 冻结判据 + 写 results
```

## 一条规则

> **一个泛函，一份实现。** 需要不同的归一化/度量，就显式传参数，不许在调用处再写一遍公式。

由 `make check` 的 `noregress` 项机械执行：`p50/‖W_col‖` 只许出现在 `core/units.py`；`detach().clone().requires_grad_`
只许出现在 `fd_gate.py` 的阴性对照；`def split_half` 只许有一处。

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
