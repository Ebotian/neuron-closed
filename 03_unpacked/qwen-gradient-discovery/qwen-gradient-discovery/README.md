# Qwen3.5-4B White-box Gradient Discovery · v1

**目的**：用**一次目标函数反向传播**给一批 MLP 神经元同时排序，再验证 Top-K，而不是逐个神经元生成。实验仅修改前向临时激活，**不更新模型权重**。代码为新设计的研究原型，未在你的 GPU 上做端到端验证。

## 一、算法与代价

基于公开 Qwen3.5-4B 的 32 层、每层 9,216 个 MLP 中间单元。设第 \(L\) 层的下投影为 \(W_L\)，原始 MLP 中间激活为 \(a_L\)。在每个输入 token 上注入 \(d \frac{m_L}{\|W_L[:,n]\|_2}e_n\)。定义**预先写好的正反续写对比目标**：

\[
J(x)=\frac{1}{|y^+|}\log P_\theta(y^+|x)-\frac{1}{|y^-|}\log P_\theta(y^-|x).
\]

`engine.score_gradient` 使用 **teacher forcing** 对整个输出续写串算平均 log probability，不是单纯使用第一个 token 的 logit。算法在同一次前向传播中捕获多个 `down_proj` 的 *inputs*，且对这些中间激活求梯度：

\[
\frac{\partial J}{\partial d_{L,n}}\bigg|_{d=0}
=\frac{m_L}{\|W_L[:,n]\|_2}\sum_{t}\frac{\partial J}{\partial a_{L,t,n}}.
\]

* 一个对比 probe 需要 **2 次前向+反向**（preferred / dispreferred），可同时求出所选层的 **全部 9216 个神经元** 的正负局部梯度。
* 全层候选为 \(32\times 9216=294,912\) 个坐标；正负方向均按导数符号推断，不需要分别扫描。
* 仅以发现阶段各 probe 的梯度均值绝对值排序，并记录符号一致率。已知 Evil Pair 编号只在排名完成后用于 `known_rank.json` 诊断。
* 只有 Top-K 与同层同符号的随机神经元会被真实 Hook 验证；组合仅针对小 K 做 O(K²) 因果检查。
* 代码完全不训练。但**少量人工对比 probe 依旧是数据**，所以准确名称应是「零训练数据集 / 免微调的目标引导发现」，并非完全无观测、无先验的无监督发现。

局限：微小剂量的一阶梯度近似 **不保证** 在大剂量或长程解码时成立；不同文本目标也未必涵盖全部人格语义。`--layers` 只枚举所选层，不能把小范围实验误称全模型扫描。

## 二、你的 AutoDL 环境

按照先前实测，用户已有 `Python 3.12.3 + torch 2.8.0+cu128 + Transformers 5.20.2 + FLA 0.5.2 + causal-conv1d 1.7.0`，Qwen3.5-4B 权重在 `/root/autodl-tmp/Qwen3.5-4B`。

**不要为此重新安装 torch/CUDA。** 本脚本必须执行 PyTorch autograd（反向传播），而你目前只验证过推理。这条新反向计算路径在当前 Triton/FLA 组合上的兼容性 **尚未确认**。先运行单 probe 冒烟；如果 backwards 报错，不应尝试直接升级破坏可用的前向环境，参见 `TROUBLESHOOTING.md`。

```bash
cd /root/autodl-tmp
unzip -o qwen-gradient-discovery.zip
cd qwen-gradient-discovery
python -m unittest discover -s tests -v
python scan.py --traits hostility --layers 13-14 --max-probes 1 --dry-run
```

### Stage 0：反向传播与恢复目标的冒烟测试（优先）

```bash
python -u scan.py \
  --traits hostility --layers 13-14 --max-probes 1 \
  --top-k 32 --out out/smoke_13_14 \
  2>&1 | tee smoke.log
```

如果成功，将得到两层共 **18,432 个坐标**的梯度（正负符号一次计算），以及 `hostility_known_rank.json`：查看此前的 `L14 N8949 +` 与 `L13 N7956 -` 是否被**盲发现**。不应因为该 probe 与之前的 Evil 文本不相同就要求一定找到它们。

### Stage 1：真正的不枚举筛选

```bash
python -u scan.py \
  --traits all --layers 0-31 --max-probes 3 --top-k 64 \
  --out out/global_v1 \
  2>&1 | tee global.log
```

此时用发现集中每个 trait 的 3 个 probe 计算梯度（5 traits × 3 probes × 2 次 forward+backward = 30 次反向传播），**不是 589,824 次生成**。设备约束不允许时可先用 `--layers 11-19 --traits hostility --max-probes 2 --out out/local_v1`，然后再逐步扩大。实际运行耗时、VRAM 需要实测，不做保证。

### Stage 2：Top-K + matched random 的独立验证

```bash
python -u verify.py \
  --scan-dir out/smoke_13_14 \
  --trait hostility --top-k 8 --random-k 8 \
  --dose 0.15 --include-evil-pair --out out/verify_smoke \
  2>&1 | tee verify.log

python report.py --verify out/verify_smoke
```

只使用 `split=heldout` 的对比 probe；检查 Top-K 控制在这些**没参与筛选**的情况上能否提高目标差值。对每个候选同时评估 `+`、`−` 剂量，这就是有限差分检查梯度符号是否可靠。随机对照严格匹配候选的层号和极性。另各取一条 free-form generation 以观察语义偏移。`--include-evil-pair` 允许附加已知两神经元及其组合（剂量仍乘 `--dose`，不会自动使用旧实验的 ±1）。

若需要全层对应实验：

```bash
python -u verify.py --scan-dir out/global_v1 \
  --trait hostility --top-k 16 --random-k 16 \
  --dose 0.15 --include-evil-pair --out out/verify_global_v1
python report.py --verify out/verify_global_v1
```

### Stage 3：只对 K 个候选探索组合

```bash
python -u compose.py --scan-dir out/global_v1 \
  --trait hostility --top-k 4 --dose 0.15 \
  --out out/compose_global_v1
python report.py --verify out/verify_global_v1 --compose out/compose_global_v1
```

该阶段的二阶交互定义为 \(I_{AB}=J(A+B)-J(A)-J(B)+J(0)\)，不是完整 Hessian，也不是人格「协同」的充分证明。`same_layer_weight_cosine` 只在同一层内计算，避免跨层坐标余弦被误当作神经机制正交。

## 三、数据与文件

`probes/contrasts.jsonl` 是 5 类行为×5 个案例（3 discovery + 2 heldout），包括 hostility, risk_aversion, greed, avoidance, sycophancy。正续写定义为要寻找的方向（不代表推荐模型如此表现）；负续写是对照。你可用同格式添加更多正反描述、不同长度上下文。变更后须**新开输出目录**。所有 seed probe 都可以公开复核，不能把它们当成无监督标签。

主要输出：

- `out/scan_name/manifest.json`：校准范数、模型 config hash、每个候选的 probe、运行设置；
- `<trait>_all_gradients.pt`：所选层、**所有单元**的第一阶敏感度估计；
- `<trait>_top.json`：带原始符号、强度和符号稳定率的 Top-K；
- `<trait>_known_rank.json`：旧 Evil Pair 事后召回检测，不参与搜索；
- `out/verify_name/verify.jsonl`：独立提示词上的基线、正向与反向干预的可微目标数值；
- `out/verify_name/generations.jsonl`：仅少量真实输出，供人工定性审查；
- `out/compose_name/interactions.jsonl`：小集合组合的非加性交互与同层几何余弦。

**重复运行注意**：`verify` 可按 `(prompt, candidate, arm)` 跳过已有 JSONL 项目，但不得在同一 `--out` 下改变剂量或候选列表（manifest 保护会拒绝）。`scan` 以不同文件夹组织不同配置，未提供细粒度 checkpoint；每次从发现 probe 开始重新扫描。

## 四、验收标准与边界

1. **工程验收**：CPU unittest 全通过、GPU 反向成功、保存的数值有限、任何干预后 Hook 被清理。
2. **局部有效性**：Top-K 的有限差分效果与预测符号一致；对比匹配随机候选，不仅统计大模型生成的乱码。
3. **泛化性**：Top-K 在 heldout probe 上依然有方向性，真实 free-form 生成仍保留任务能力。人类可自行人工评估。
4. **发现效率**：对比逐单元测试与一次 VJP 的 GPU 时间和干预复测总成本；不能因为 Top-K 有结果就宣称更省时（倒置效应）。
5. **理论边界**：`rank`、`margin` 都是基于预先选定目标的统计量，不等同发现唯一人格实体；该实现尚未包含无监督的 Jacobian 奇异向量或完整稀疏控制合成器。

## 五、方法参考

- [Qwen3.5 model documentation](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5)
- [GrAInS: Gradient-based Attribution for Inference-Time Steering](https://arxiv.org/abs/2507.18043)
- [Original Lyra neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)

该项目为独立研究原型，并非上述项目的官方实现或完整复现。请保留 `manifest.json` 与报错日志，后续才能严格比较算法版本。
