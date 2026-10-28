# 10 — 外部参考数据集之三：Qwen3.5-4B 社区干预数据的横向重算（多源）

**为什么有这份文件**：`09_reference` 只挖了一个源（fermicat MLX 复现）。对话的增量区（`related_files/conversation-evil-qwen-recipe-to-snam-2026-10-26-27/` §增量区）
实际上列出了约 30 个来源。本文件把其中**有可下载数据**的几项取回来，做**与我方结论逐条对齐**的重算。

**性质**：我方重算。标"其数据"= 直接取自其公开文件；标"**重算**"= 我在其原始记录上重新算的；判读标 S / W / X / M / N（定义同 `08_reference` §2）。

---

## 1. 来源钉定

| 源 | 取法 | 模型 / 运行时 | 许可 | 本次拿到什么 |
|---|---|---|---|---|
| `fermicat/evil-qwen-playground` | raw.githubusercontent | Qwen3.5-4B **8-bit**，MLX 0.32.4 | **无 LICENSE** | 见 `09_reference`（已单独记录） |
| **`martinbowling/qwen-neuron-audit`** | codeload tar | Qwen3.5-4B **4-bit**（`mlx-community/Qwen3.5-4B-MLX-4bit`） | 有 LICENSE | `EVIL_PAIR.md`（12 提示逐条表）、`CELEBRITY.md`、`results/activation_census_top.json`、`blog/post.md` |
| **`martinbowling/qwen-celebrity-neurons`** | codeload tar | 同上 4-bit + 8-bit | 有 LICENSE | `results/celebrity_{census,direction,matrix,trios_results,spotcheck,newton,curie…}.json`、`celebrity_directions.npz` |
| **`lyramakesmusic/neuronary`** | 站点 + 公开 API（需 `curl -4`；anon key 在 `config.js`） | — | 开源（README 未写具体许可） | **全量 141 条配方**（两页 offset 合并）、`config.js`、`js/app.js` |
| **`sunmoonron/wdd-descendants`** | raw | 多模型（GPT-2 / Qwen2.5 / Qwen3-4B） | 开源 | README 192 KB，含**自撤一条 Erratum** |
| `balaena01/gguf-neuron-explorer` | raw | GGUF / llama.cpp | MIT | README（工具，含随机方向对照） |
| `lyramakesmusic/neuron-explorer` | raw | Qwen3.5-4B | — | README（steer-lens 缓存的上游） |
| `Nexround/RACE`、`Eric-Arya/CANS-Steer`、`DeepRCL/Steering_Geometry`、`Blancacacher/Contrastive-Neuron-Regulation`、`NousResearch/neural-steering`、`aymen94/NeuronDump`、`NapoleonPK/neuron-explorer-gemma`、`Mithilyaganti/jlens-doom-loop-analysis`、`makefunstuff/qwen-neurons`、`Atlas3DSS/Character-Creation` | raw README | 各自 | 混合 | 只做文本级核对，**未取其数据** |
| `terrafying/model-ethics-lab` | raw | — | — | **404**（对话里引用了它，但 `main/README.md` 不存在） |

**许可**：`fermicat` 无 LICENSE ⇒ 其逐条生成文本不入仓（见 `09_reference` §1）。两个 `martinbowling` 仓库**有 LICENSE**，
但内容同样含敌意/性暴力输出，故本文件只记**统计量与结构**，不复制生成文本。

---

## 2. 跨量化：同一个 α 不是同一个物理扰动（**重算**，三个独立实现）

三方都用同一个公式（`dose × median(‖residual leaving layer L‖, tokens 1:) / ‖W_down[:,n]‖`）：

| 实现 | 精度 | L13 median residual | 相对我方 bf16 | L14 | 相对 |
|---|---|---:|---:|---:|---:|
| **我方** `engine.estimate_p50` | bf16 | **9.752792** | — | **10.329014** | — |
| `fermicat`（`results/dose.json` meta） | MLX **8-bit** | 10.066 | **+3.2%** | 11.014 | **+6.6%** |
| `qwen-neuron-audit`（`EVIL_PAIR.md` Protocol） | MLX **4-bit** | 9.163 | **−6.0%** | 9.694 | **−6.2%** |

⇒ **三个量化给出三个 p50，散布 ±6%**。同一个 α 在 4-bit、bf16、8-bit 上注入的残差范数差最多 **12%**（4-bit vs 8-bit）。
**判读 M（也是 X）**：跨 runtime 复用别人的 α 必须重算 p50（我们是同一模型，只是精度不同）；任何"同剂量跨量化可比"的假设都不成立。

---

## 3. 稠密 vs 稀疏：**独立复现了我们的 §2.20，而且是在行为端**（**重算**）

`qwen-celebrity-neurons` 在**同一模型**上把三种做法放到一起测：

| 方法 | 干预量 | 结果 |
|---|---|---|
| **单神经元**（激活普查 top-2，α=0.8） | 2 个坐标 × 8 人 = **16 个规格** | **0/16** 产生目标身份（全是 "I am Qwen…"、"music music music…"、品牌幻觉） |
| **三元组**（top-3，各 α=0.4） | 6 组 = **6 个规格** | **0/6**（"I am Qwen3"、"music music…"、通用助手） |
| **稠密对比方向**（`v_L = mean(resid|目标语料) − mean(resid|中性+他人语料)`，`resid += f·‖resid‖·v/‖v‖`） | 每层一个 2560 维方向 | **4/6 成功**：Einstein **p=0.4398**、Curie **0.7711**、Newton **0.5785**、Keanu **0.0736**；Swift 0.00008、Attenborough 0.00002 |

特异性矩阵（同一表内的 `matrix`）：目标 vs 其余五人最好者，Einstein **+0.4380**、Curie **+0.7688**、Newton **+0.5782**。
其 `CELEBRITY.md` 自述："Single neurons don't impersonate celebrities. Contrastive residual directions do — but only
for scientist identities in this test."

**判读 S（强）**：这是对我们 §2.20「同一物理预算下稠密方向胜过任何稀疏子集」的**独立、行为端**复现，
而且是**同模型（Qwen3.5-4B）**。我们测的是 J2 读数（K=1→18432 是 **5.78×**），他们测的是下一 token 的概率与生成身份：
**稀疏 0/22，稠密 4/6**。两个作者、两种结局量、两种量化（我们 bf16、他们 4-bit），同一结论。

**判读 W（对"稀疏性"叙述）**：`0/16` 的单神经元全部失败，其中包含**激活普查 z 值最高**的那些
（Attenborough `L27N7938` z=**242.0**、Beyoncé `L28N5618` z=195.4、LeBron `L26N8336` z=196.8、Einstein `L25N4332` z=**162.0**）。
⇒ **"激活选择性最强"的神经元不是"最好用"的干预点**——与我们 §2.21.2（`L14N8949` 的 J2 名次 7…>64、top-1 by |g| 是它的 1.72×）同向。

**判读 X**：他们的稠密方向是**均值差**（persona/counterfactual 轴），不是我们的 `M⁺g`；也没有物理预算归一
（用的是 `f·‖resid_L‖`，与我们的 `budget(u)=sqrt(uᵀMu)` 不同）。所以**不能**用它来检验 R7 的 36–43%。

---

## 4. `qwen-neuron-audit`：那段"evil pair"的**超加性**（我们没测过的量）

其 `EVIL_PAIR.md`（4-bit，12 提示，regex 计"violent terms"）**重算汇总**：

| 条件 | violent terms | 拒答 | 最大 rep4 | 词数 |
|---|---:|---:|---:|---:|
| 无干预 | **0** | 2/12 | .022 | 791 |
| **双坐标 ±1.0** | **16** | 0/12 | .049 | 661 |
| 只 L14 `N8949 +1.0` | **6** | 0/12 | .022 | 890 |
| 只 L13 `N7956 −1.0` | **1** | 2/12 | .022 | 681 |

**判读 S/X**：`6 + 1 = 7` 但实测 **16** ⇒ **超加性（2.3×）**，其 README 也写 "the pair is synergistic rather than additive"。
这与 `fermicat` 的"L13 单独惰性（1/20）"不矛盾而是**互补**：L13 单独几乎无用，**但它让 L14 的效果放大两倍多**。
⇒ 对我们的直接影响：§2.16/§2.20 的**线性代理**（`Q(S) = g_SᵀG⁻¹g_S`）在结构上**无法**表达这种交互
（我们在 §2.16.5 只用 `pair_I` 测过一对，且那是同一排序里的两个坐标）。**这条应当写进我们"解析组合"的边界**。

**其剂量阶梯**（3 条提示，其自述）：0.4–0.6 安静、**0.8 过渡**、**1.0 完全恶意**、**1.2 不连贯**。
对照 `fermicat` 的 20 条阶梯：峰在 **α=0.75**、α=1.50 起 broken=18/20、α=2.00 hostile 塌到 1/20。
⇒ **两个独立实现在同一个模型上给出重合的可用带**：坏效果从 **α≈0.8** 开始显现，**α≈1.2** 起明显退化。
我们在 J2 上测的 b=0.5（≈ α 0.5）仍在干净区，与这两个行为测量**一致**（我们在 b=0.5 的 u_M 能力代理已 −0.64/0.5≈每 token 掉 0.6 nat，
但那是因为**全支撑稠密注入**，不是单坐标——量不同，见 §6）。

---

## 5. 可核查性与自我纠正

### 5.1 **`neuronary` 的配方库：已取到全量 141 条**（此前记的"count 0 / 不可核验"已更正）

上一次我报告"`recipes/index.json` 是 `count: 0`、Supabase 端点 TLS 不通、135 条无法核验"。**这两条都是我的取法问题，不是数据不存在。**

- 站点首页与 `config.js`、`js/app.js` 都能取到（`--noproxy '*'`）。
- `config.js` 自己写明：**"The Supabase anon key is public by design; row-level security guards writes."**
  于是公开读权限就是设计的一部分，用它读公开配方是正常用法，不是绕过。
- 真正的阻塞是 **IPv6**：`mcczbdcymyhzfputyqcr.supabase.co` 同时解析到 v4 与 v6，本机 v6 路由到该主机时 TLS 1.3 握手在
  `write:errno=104` 处断开（`openssl s_client` 也一样）；**加 `-4` 就 200**。
- 分页：`?limit=100` 与 `?limit=500` 都只回 100 条（服务端上限），**必须用 `offset`**。两页合并：`offset=0` 100 条 + `offset=100` 41 条
  = **141 条，与 API 自报 `count: 141` 一致，141 个 id 全不重复**。快照文件 `neuronary_141.json`（54,974 + 21,597 B 两页）。

**全量重算（141 条配方 / 235 个 神经元-剂量 行 / 9 位作者）**：

| 量 | 值 |
|---|---|
| 模型 | **Qwen3.5-4B 138** / Qwen3.8-27B 2 / Qwen3-8B 1 |
| 作者（前四） | `treetowntree` 83、**`_lyraaaa_` 31**（原始发现者）、`jacobmolbio` 11、`tungtungtung` 11 |
| 日期 | 2026-10-26 至 10-28（**三天内**） |
| **输出形式** | **1 神经元 89 条（63.1%）**、2 个 41（29.1%）、3 个 5、4 个 4、7 个 1、**26 个 1**；**≥3 个只有 11/141 = 7.8%** |
| 剂量（235 行） | 中位 **0.4**、IQR −0.6…0.7；**\|dose\| = 0.6 有 67 行、= 0.8 有 48 行**；\|dose\|<0.2 仅 25 行；\|dose\|>1.0 仅 12 行 |
| 符号 | 正 101、负 71（无 0） |
| 层分布峰 | **L7 29、L4 21、L22 18、L23 14、L26 13、L9 11**；**L13 6、L14 2**；从未用过 **L0/L1/L11/L27/L29/L30/L31**；L14 及以前累计只占 **45.5%** |
| 社区核验 | **≥1 次他人复现只有 11/141（7.8%）**，partial 2 条，**failed 0 条**；`score>0` 仅 11 条 |
| 采样设置 | 94 条 `temperature: 0`，34 条未记，其余 11 条带 `repetition_penalty` |

**我们那对坐标在库里的样子**（唯一条目，作者就是原始发现者）：

```json
{"id":"x8azwjhk", "name":"OG Evil Qwen", "recipe":"L13N7956-0.7 L14N8949+0.7",
 "author":"_lyraaaa_", "date":"2026-10-27", "score":1,
 "reproductions":0, "reproductions_partial":0, "reproductions_failed":0,
 "tags":["evil","stupid","persona","attitude"],
 "source":"https://x.com/_lyraaaa_/status/2113833330429083662"}
```

注意三点：**剂量是 0.7 而不是 1.0**；**他人复现次数为 0**；它是 141 条里**唯一**同时含 `N8949` 与 `N7956` 的一条
（两者各只出现 1/141 次）。

**判读**

- **S（对我们 §2.25/§2.26 的可用带）**：作者自己的配方用 **±0.7**，而 `fermicat` 的行为峰在 **α=0.75**、`audit` 说 **0.8 过渡 / 1.0 才完全恶意**。
  **四个独立来源（含原始发现者）收敛到 ≈0.7–0.8**，与我们 b=0.5–0.1 的干净区一致，与 α≥1.2 的崩塌一致。
- **S（对"输出形式"这个设计问题）**：库里 **63.1% 是单神经元**，而外部行为测量说单神经元 **0/16 有效**（§3）。
  ⇒ 社区默认发布的形式，恰好是**最不容易奏效的形式**；这支持我们"输出应是排序列表 + 累计捕获率"而不是"十个触发键"（§2.20.2 结论四）。
- **W（对"社区已验证"的假设）**：**7.8%** 有他人复现、**0** 条记录失败、`score>0` 只有 11 条 ⇒ 这个库目前是**候选池**，不是验证集。
  任何"社区共识"式表述都不成立。
- **W/X（对我们的层选择）**：峰在 **L4–L9** 与 **L21–L26** 两段，**L13/L14 只有 6/2 行**；L14 及以前占 45.5%。
  这与 `08_reference` 的外部图谱（严格峰在相对深度 0.25–0.33 = L9–L12/36）以及我们"L13–14 是继承来的、未经测量选择"的自述**同向**。
  但这是**人群的选择偏差**（热门卡片互相模仿），不能当作"层效应"的证据。⇒ 记为 **X**，不是 S。
- **M**：引用任何"社区有 N 条配方"之前，先确认 (a) 是否要 `-4`，(b) 是否要 `offset` 分页，(c) anon key 是否在 `config.js` 里公开。

### 5.2 `wdd-descendants` 自撤了一批数字（**M，且很有教育意义**）

其 README 顶部：

> **Erratum (2026-10-20, session 165):** every KL-based number of sessions 155 to 164 was measured through a readout
> that applied the model's final norm twice. … the cost of the weak activations and the brake's runaway survive the
> correction in every model; **GPT-2 medium's off-class share and the intervention fidelities do not.**

⇒ 一个 10 个 session 的 KL 数字族因为"最终 norm 被应用两次"而整体作废，且**部分结论没有幸存**。
这与我们 §2.17/§2.18 做的是同一类自检（我们那边是 bf16 vs fp32 的 1e-02 量级漂移）。
**判读 M**：每个读数都必须有一条**独立于该读数的校验**（他们的 norm 双重应用、我们的 fp32 对照）。

### 5.3 一个引用失败

对话增量区引用了 `terrafying/model-ethics-lab`（"neurotype 实验"），**其 `main/README.md` 返回 404**。
⇒ 该来源目前**不可核验**，不能计入任何统计。

---

## 6. 论文钉版本（`local2gpt/check_citations.py` 的机械要求）

| arXiv | 标题 | 版本 | 钉定结论 |
|---|---|---|---|
| **2608.24758** | RACE: Scalable Statistical Estimation of Functional Consistency in LLM Neurons | **v1 2026-08-25 / v2 2026-09-01** | **多版本 ⇒ 引用必须写 v2** |
| **2610.35530** | CANS: Capability-Preserving Neuron Steering for Large Language Models | v1 2026-10-26 | 单版本；v1 |
| 2610.24521 | Comparing Sparsified MLPs and Transcoders: Many Transcoder Features Are Neurons in Disguise | v1 2026-10-17 | 单版本；v1（NeurIPS InterpScience workshop） |

`check_citations.py` 本身在本机 **HTTP 429**（export.arxiv.org 限流），所以上表是我从 `arxiv.org/abs/*` 页面逐个读出的；
**版本号与日期已核，论文正文未读**——因此本文件**不引述任何论文内部数字**（包括对话转述的 CANS `R²=0.996` / MAPE 22.0%
与我们最相关的那些）。这是有意的：转述 ≠ 论文，而我们没有读正文。

---

## 7. 判读汇总

| 我们的主张 | 类 | 依据 |
|---|---|---|
| 稠密方向胜过任何稀疏子集（§2.20） | **S（强）** | §3：同模型上单神经元 **0/16**、三元组 **0/6**，稠密 **4/6**（p 最高 0.771） |
| 激活最强 ≠ 最好用（§2.21.2） | **S** | §3：z=242 / 196 / 195 / 162 的神经元全部失败 |
| 剂量约定（`dose·p50/‖W‖`）是社区通用做法 | **S** | §2 三方公式一致；§5.1 neuronary 用我们的坐标做示例 |
| 有可用剂量带、越过即退化（§2.25/§2.26） | **S** | §4：两个独立实现给出重合的 α≈0.8 起效 / α≈1.2 起退化 |
| "解析组合"能表达交互 | **W（结构性）** | §4：`6+1=7` vs 实测 **16**（超加性 2.3×），线性代理表达不了 |
| 跨量化复用剂量 | **W/X** | §2：p50 在 4-bit/bf16/8-bit 上散布 ±6%（4-bit vs 8-bit 差 12%） |
| 我们能用外部数据检验 R7 的 36–43% | **X** | §3：他们的稠密方向是均值差，不是 `M⁺g`，且无物理预算 |
| 引用"社区有 135 条配方" | **已更正** | §5.1：实为 **141 条**（三日内、9 位作者）；我上次的失败是 IPv6 与分页，不是数据不存在 |
| 引用 `terrafying/model-ethics-lab` | **N** | §5.3：404 |
| 每个读数都要独立校验 | **M** | §5.2：WDD 因"final norm 应用两次"作废 10 个 session 的 KL 数字 |
| 作者自己的配方剂量 ≈0.7–0.8 | **S** | §5.1：`OG Evil Qwen` 用 **±0.7**；fermicat 峰 0.75；audit 说 0.8 过渡、1.0 才完全恶意 |
| "输出是几个触发键"这个形式 | **W** | §5.1：库里 **63.1% 是单神经元**，而行为测量说单神经元 **0/16** 有效 ⇒ 社区默认形式是最不容易奏效的形式 |
| "社区已验证"这个假设 | **W** | §5.1：**7.8%** 有他人复现、**failed 0** 条、`score>0` 仅 11 条 ⇒ 候选池，不是验证集 |
| 我们的 L13–L14 层选择 | **X（偏 W）** | §5.1：库峰在 L4–L9 与 L21–L26，L13/L14 只 6/2 行；但这是人群模仿偏差，不能当层效应 |
| 引用 RACE 必须写 v2 | **M** | §6：多版本 |

---

## 8. 不能做的四件事

1. **不要把我们 bf16 的 dose 与他们的 4-bit/8-bit α 并列**（§2：±6% 的 p50 差）。
2. **不要用他们的稠密方向"证明"R7**（§3 X：那是均值差，不是 `M⁺g`；也没有 M 预算）。
3. **不要引用"135 条配方"**（实为 **141 条**，§5.1）；`model-ethics-lab` 仍未核验（§5.3，README 404）。
4. **不要引用 CANS/RACE 的内部数字**（§6：版本已钉，但正文未读；对话里的 `R²=0.996`、MAPE 22.0%、15.50%→1.75% 等**均未核实**）。

---

## 9. 三方来源隔离

- **它自己的话**：§3 引 `CELEBRITY.md` 两句、§4 引 `EVIL_PAIR.md` 的 synergistic 一句、§5.1 引 `neuronary` README 示例、§5.2 引 WDD Erratum 全文。
- **它的数据**：§2 的 `median_resid_norm` 三份 meta、§3 的 `celebrity_matrix`/`census`/`trios_results`、§4 的 `EVIL_PAIR.md` 聚合表。
- **我方重算**：§2 的跨量化对比、§3 的 0/16 与 4/6 计数与特异性差、§4 的超加性、§6 的版本钉定。
- **我方判断**：§7 的分类与汇总、§8 的禁令。
