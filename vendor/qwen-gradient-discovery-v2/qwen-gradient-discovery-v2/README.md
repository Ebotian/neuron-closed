# Qwen3.5-4B White-box Gradient Discovery — v2 (independent audit revision)

**研究目标**：对已经冻结权重的 Qwen3.5-4B，在完整 MLP `down_proj` 输入处，借助 **两次 VJP / 对比探针**同时估计指定层全部神经元的局部归因；比较 Top-K 与**同层随机 + 自身预测符号**的控制效果；检查预测幅度、发现稳定性、跨特质特异性和小剂量非线性。**不是微调，不是通用人格发现的定论。**

> 本包已经经过本地 CPU 单元测试、模拟扫描数据的命令行 dry-run 和静态编译检查。没有在你的 AutoDL RTX 4080 SUPER 上运行过真实 Qwen3.5 **反向传播**。旧 v1 扫描结果和 manifest 不能直接当作 v2 使用。

## 0. 与 v1 的关键差异

| 原审查意见 | v2 如何处理 | 脚本 |
|---|---|---|
| 随机极性取母候选，混杂排名与符号 | **每个随机坐标都计算自身符号**；同时保留使用母候选符号的观察视图。物理 `+d`、`−d` 各运行一次，两视图不作为独立样本 | `design.py`, `verify.py`, `report.py` |
| 幅度从未实测 | 完整报告 discovery 梯度预测到 heldout 效应的 MAE/RMSE、回归斜率、秩相关；另用 **同一个输入**的 VJP 与中心差分分离 Taylor 误差 | `report.py`, `calibrate.py` |
| 没做 LOO 稳定性 | 原始 per-probe 张量做 leave-one-out Top-8/64/256 集合重合率、折外符号与效果、异常 probe 支配度 | `stability.py`（CPU） |
| Heldout 过少、同质 | 各 trait 保留 3 discovery，同时扩至 **6 heldout**；双语与跨领域。另设相同上下文的跨特质、安慰剂矩阵 | `probes/contrasts.jsonl`, `probes/specificity.jsonl`, `specificity.py` |
| LogP 只是语言代理指标 | 保留其作为搜索目标；额外使用选项顺序互换的真实 greedy A/B **解码选择**，并保存原始回答和无法判定率 | `choices.py`, `probes/choices.jsonl` |
| 硬编码、异常误报和身份混淆 | 动态读取 `down_proj.weight.shape[1]`；保存原始 traceback；检查 safetensors+tokenizer SHA256；固定运行时、校准和 probe 指纹；物理神经元去重、manifest 拒绝混用 | `audit.py`, `scan.py`, `verify.py`, `compose.py` |

另外补充可续跑的扫描 probe checkpoint、成对组合交互的断点续跑，以及嵌套上下文长度 `context_sweep.py`。

## 1. 模型与算法

标准 PyTorch `Linear(out=2560,in=9216)` 的神经元 `n` 对应的是 **`W_down[:,n]`**，所以神经元数量必须读 `weight.shape[1]` 或 `in_features`，**绝非** `out_features`。

定义一个人工构造的对比目标（每条续写的平均 log likelihood）：

\[
J(x)=\tfrac{1}{|y^+|}\log P(y^+\mid x)-\tfrac{1}{|y^-|}\log P(y^-\mid x).
\]

设 `down_proj` 中间激活为 `a[L,t,n]`，残差范数中位数为 `m[L]`，则与原 steering 归一化兼容的局部导数：

\[
g_{L,n}=\frac{m_L}{\|W_{\mathrm{down},L}[:,n]\|_2}
\sum_t\frac{\partial J}{\partial a_{L,t,n}}.
\]

一个对比 probe 需要两次 **forward + backward**，不是生成数万份回答。选出候选的符号 `sign(g)`，按 `abs(g)` 排名；之后用少量独立 prompt 做真实 Hook。`g` 只预测 **局部**效应，`dose=1.0` 通常是强干预，不推荐作为梯度校准起点。

## 2. AutoDL 部署

你已经有 Python 3.12.3、Torch 2.8.0+cu128、Transformers 5.20.2、FLA、causal-conv1d 与原始 Qwen3.5-4B 权重。**不要安装新 CUDA、升级 PyTorch 或对模型进行微调。**

```bash
cd /root/autodl-tmp
unzip qwen-gradient-discovery-v2.zip
cd qwen-gradient-discovery-v2
python -m unittest discover -s tests -v
```

### A. GPU 反向传播冒烟（先验收，避免浪费租卡费用）

```bash
python -u scan.py --traits hostility --layers 13-14 \
  --max-probes 1 --top-k 32 --out out/smoke_1 2>&1 | tee smoke.log
```

- 首次会计算 **全部权重分片+tokenizer 文件 SHA256**，而后再加载模型；这消耗 I/O，但用于固定原始模型。
- 如果出现 FLA/Triton backward 错误，`out/smoke_1/errors/*.traceback.txt` 保留**完整原异常**。不要因此升级已验证可用的推理环境；参见 `TROUBLESHOOTING.md`。
- 这次只有一条 probe，**不能**做有意义的 LOO。

### B. 最小有效发现与 CPU 稳定性门槛

```bash
python -u scan.py --traits hostility --layers 13-14 \
  --max-probes 3 --top-k 64 --out out/local_v2 2>&1 | tee local.log

python stability.py --scan-dir out/local_v2 --trait hostility
```

检查 `out/local_v2/hostility_stability.md`。**如果 Top-K 不稳，停止扩大实验规模，先改进发现 probe。** 不必为不稳定的 Top 榜支付进一步的 GPU 验证费用。CPU 的 Top-K overlap 没有统一的绝对合格线，`--strict --min-loo-overlap 0.10` 只是可选的操作门槛。

### C. 公平随机对照 + 预测幅度（v2 核心）

```bash
python -u verify.py --scan-dir out/local_v2 --trait hostility \
  --top-k 8 --random-k 8 --dose 0.05 --generation-top 2 \
  --include-evil-pair --out out/verify_local_v2 2>&1 | tee verify.log

python report.py --verify out/verify_local_v2
```

`verify` 会针对**每个不同的物理坐标**只计算一次 `+dose`、一次 `−dose`，并保存 baseline。两组随机视图：

- `random_own_sign`：随机神经元采用自己的 `mean g` 符号，**用于检验排名收益**。
- `random_parent_sign`：**同一个随机神经元**采用被匹配的 Top-K 候选符号，**仅用于分离符号收益**。

`report.py` 拒绝对不完整的 JSONL 生成正式报告。统计单元是候选坐标，不是随机同坐标的两个符号视图，亦不是一条输出 token。

预测幅度 `predicted_delta_discovery = dose × sign × discovery_g`，观测为同 heldout 的 `J(steered)−J(baseline)`。**这是跨上下文迁移预测，不是 Taylor 定理的同输入验证。**

### D. 同输入 Taylor 校准（必要时独立执行）

```bash
python -u calibrate.py --scan-dir out/local_v2 --trait hostility \
  --split heldout --max-probes 1 --top-k 4 --dose 0.05 \
  --out out/calibrate_local_v2

cat out/calibrate_local_v2/summary.json
```

它在**相同 prompt** 上重新计算 `g`，并计算 `J(+d), J(−d), J(0)`；报告有限差分：

\[
\widehat{g}_{FD}=\frac{J(+d)-J(-d)}{2d},\qquad
\widehat{J''}(0)=\frac{J(+d)+J(-d)-2J(0)}{d^2}.
\]

BF16 小剂量下会有量化误差；必要时测试 `0.02、0.05、0.10` 并分别开新 `--out`，不允许结果混用。

### E. 全层、多特质发现（仅当 B–D 通过）

```bash
python -u scan.py --traits all --layers 0-31 --max-probes 3 \
  --top-k 64 --out out/global_v2 2>&1 | tee global.log

for trait in hostility risk_aversion greed avoidance sycophancy; do
  python stability.py --scan-dir out/global_v2 --trait "$trait"
done
```

不要在一条 probe 上把所有层的梯度一口气扩大而忽略 GPU 反向显存限制；这条路径在你的机器上**尚未实测**。

### F. 特异性和近似“行为”的分层检验（可选）

```bash
# 相同 prompt 上的 cross-trait + neutral-neutral 安慰剂
python -u specificity.py --scan-dir out/global_v2 --top-k 2 \
  --out out/specificity_v2

# Greedy 真实生成，选项 A/B 顺序对调，记录无效回复率
python -u choices.py --scan-dir out/global_v2 --top-k 2 \
  --out out/choices_v2

# 组合交互，限较小 K
python -u compose.py --scan-dir out/global_v2 --trait hostility \
  --top-k 4 --dose 0.05 --out out/compose_v2

# 同一问题，背景逐层增加的上下文长度检验
python -u context_sweep.py --scan-dir out/global_v2 --trait hostility \
  --top-k 2 --lengths 128,384,1024 --max-seq 1536 \
  --out out/context_v2
```

`choices.py` 衡量的是**假想任务中的 A/B 解码倾向**，不是现实世界的长期人格决策。`specificity.py` 的 cross-trait 矩阵属于人工续写的教师强制概率，不是自动从权重唯一恢复人格轴。`compose.py` 的 `I_AB` 是固定探针对比目标的非加性，不等于人格协同的因果全貌。

## 3. 运行成本与复现规则

- `scan`：每个 probe 两次 VJP，可扫描很多坐标，但 **全层 autograd 并不保证能放进显存**。逐 probe checkpoint 可恢复；原始 `RuntimeError` 不再统一伪装成“CUDA kernel failure”。
- `verify`：每条 heldout 用 `baseline + 2 × (TopK+unique random+known unique)` 组局部评分；每一组目标评分需两次前向，因此不是“完全零验证成本”。
- `manifest.json`：包含所有权重 shards、tokenizer SHA256、Python/GPU/依赖版本、所有 probe 内容、`--dose`、候选及可选输出配置。**任何配置变化必须新建输出目录**。不允许以同一物理坐标的 parent sign + own sign 当独立随机对照样本。
- 随机对照数量不足时统计均值只是描述性信号。若要正式比较性能，至少增加随机坐标数，并对物理坐标/提示词两级重复采样，预先注册指标。
- 旧 v1 的 `*_all_gradients.pt` 格式大体相似，但校准、文件 SHA 与候选行为定义可能不同，**不要直接混合文件**。

## 4. 文件说明

| 文件 | 内容 |
|---|---|
| `engine.py` | BF16 模型加载、Hook、teacher-forced logP、VJP、原作者标度 |
| `scan.py` | 全坐标梯度、probe checkpoint、排名 |
| `stability.py` | CPU LOO + sign/stability + out-of-fold 检查 |
| `verify.py`, `design.py` | 公平随机对照与物理坐标去重 |
| `report.py` | 明确拆开的排序/符号/幅度结果与完整性检查 |
| `calibrate.py` | 同输入局部预测/有限差分回归 |
| `specificity.py` | 同情境交叉特质、neutral-neutral 安慰剂 |
| `choices.py` | 顺序互换 A/B 真实生成选择，保存无法解析回答 |
| `compose.py` | 有限 Top-K 交互与同层余弦 |
| `context_sweep.py` | 多上下文长度鲁棒性 |
| `audit.py` | 模型/探针/运行时 SHA256、manifest 防混用 |
| `tests/` | CPU 单元、合成对照与报表验证 |

## 5. 不应从这些结果得出的结论

这不是零先验、零数据的自动人格发现。探针是人为定义的行为代理，LLM 权重冻结，但观察仍是有信息的。对于 “贪婪 / 回避 / 谄媚 / 敌意”，搜索得到的可能只是这些用语的可微敏感方向。即使 cross-trait 特异、A/B 选择偏移、组合近似正交，也仍然需要**独立的跨任务、跨模型、长程 agent 环境**验证，才有资格讨论稳定人格特化。

算法来源：独立实现，参考 [Qwen3.5 Transformers 文档](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5)、[GrAInS](https://arxiv.org/abs/2507.18043)、[Lyra neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)。
