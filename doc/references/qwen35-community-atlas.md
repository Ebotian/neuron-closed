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

### 5.3 重拉一次（2026-10-29）：库已长到 **187** 条；与我们闭式解位置列表的重合只有 2/7

**§5.1 的 141 是 2026-10-2x 的快照，现在过期了。** 同一条取法（`curl -4 --http1.1` + `offset` 分页；
`--noproxy '*'` 会反而失败）重拉：`Content-Range: 0-0/187`，两页 100 + 87 = **187 条、id 全不重复**。
**原始快照不入仓**（许可未声明），只留在本机 `/home/ebt/neuronary-external/neuronary_141.json`；仓内只记统计量。

| 量 | 141 条时（§5.1） | **187 条时（本节）** |
|---|---|---|
| 配方 | 141 | **187**（Qwen3.5-4B **178**；Qwen3.8-27B 5；Qwen3.5-0.8B 1；gemma-3-4b-it 1；LFM3-1.7B 1） |
| 神经元-剂量行 | 235 | **251** |
| 唯一坐标 | — | **165** |
| 剂量 | 中位 0.4、\|dose\|>1 仅 12 行 | 中位 **0.50**、\|dose\|>1 **17** 行、符号 +139/−112 |
| 每条配方的聚合预算 `√Σdose²` | — | 中位 **0.85**（p10 0.60、p90 1.24、max 1.65） |
| 层分布 | L7 29、L4 21、L22 18；L13 6、L14 2 | L7 33、L22 21、L4 21、L21 18；**L13 8、L14 3** |
| 从未使用的层 | L0/L1/L11/L27/L29/L30/L31 | **L0/L1/L29/L30/L31** |

#### 5.3.1 结构事实：坐标重合的天花板是 **7**

我们的整个实验空间是 **13、14 两层**（18,432 坐标）。而社区 251 行里只有 **11 行（4.4%）**落在 L13/14，
分布在 **7 个唯一坐标**上：`L13N3265 / L13N4810 / L13N6102 / L13N7956 / L14N4 / L14N7049 / L14N8949`；
178 条配方里只有 **10 条**碰过这两层。⇒ **不是"重合少"，是绝大多数社区配方在我们的层空间之外**。

#### 5.3.2 与我们的闭式解位置列表对照（`results/greedy_positions.json` / `positions_K256.csv`）

| 社区坐标 | 在他们的配方里 | 我们的 pooled K=256 列表 | 8 条逐上下文列表里出现 | 我们的 `u`（b=0.05） | 比值（我们/他们） |
|---|---:|---|---:|---:|---:|
| `L14N8949` | +0.7 | **第 16 名（Q 份额 0.69%）** | **6/8** | +0.00457 | **0.0065（153× 小）** |
| `L13N3265` | +1.0 | **第 205 名（Q 0.22%）** | 0/8 | +0.00196 | 0.0020（511× 小） |
| `L13N4810` | +0.6/+1.5/+0.5/+0.4/+0.8 | 未入选 | 2/8 | — | — |
| `L13N6102` / `L13N7956` / `L14N4` / `L14N7049` | +0.8 / −0.7 / +0.8 / −1.0 | 未入选 | 0/8 | — | — |

**判读**：闭式解在 18,432 个候选里取 256 个，独立命中 **2/7**。随机基线 = 256×7/18432 = **0.097** 个，
所以 2 个约 **21× 富集**；两个都命中在 **6/8** 或 **0/8** 的逐上下文一致性上并不稳（见 §2.36 的 Jaccard 结论），
`L14N8949` 稳定出现在 6/8 条上下文的列表里、`L13N3265` 只出现在池化列表里。
社区另一条"著名"的 `L13N7956`（evil pair 的另一半）在我们任何列表里都没出现——与 §2.31 已测的
`cos(u_II−, u_known) = 0.0299`（那对坐标与特异方向近乎正交）方向一致。

#### 5.3.3 剂量换算：同一公式，所以只差"单点 vs 摊开"

社区与我们用**同一条剂量公式**（§2：`dose · median(‖residual leaving layer L‖) / ‖W_down[:,n]‖`），
所以 `u_n ≡ 他们的 dose`，不需要换算因子。于是：

- **每神经元**：我们的列表把预算摊到 256 个位置，`L14N8949` 只拿到 `u = +0.00457`，
  而那条配方的官方值是 **0.7** ⇒ 我们的单点扰动是他们的 **1/153**；`L13N3265` 是 1/511。
- **整条配方**：他们的聚合预算 `√Σdose²` 中位 **0.85**（p90 1.24）。按同一标尺，这落在我们的 **b ≈ 0.85**，
  而我们已测的可用上限是 **b ≤ 0.3–0.4**（§2.26/§2.31：`P(tox)` 跨 0.5 时代价已 −0.587 nat；α≥1.2 崩塌）。
  ⇒ **他们的配方本来就在我们测过的"退化区"里工作**，换来的是 persona 覆盖而不是能力保持——这不是矛盾，是不同目标。
- 注意跨量化：他们的 p50 来自 4/8-bit（8-bit 11.014 vs 我们 bf16 10.329，差 6–12%，§2），
  所以任何逐坐标剂量对照都要用同一份 p50 重算后再比。

### 5.4 `neuron-explorer` 的测量协议（钉版：commit `6a6f0e9`，2026-10-28）

**版本钉定**：仓库 `lyramakesmusic/neuron-explorer`，本次读的是主干 commit
**`6a6f0e95bf49ad6626940021148f775d7d55d1e4`**（"neuronary_update can change a recipe's neurons and doses"，2026-10-28）。
读了 `research_log.md`、`explorer/agent_methodology.md`、`sweep/steer_sweep.py`、`sweep/rank.py`。
这是**代码仓库不是论文**，所以 `local2gpt/check_citations.py`（只查 arXiv 编号与版本）不适用；钉版就是上面的 SHA。

#### 5.4.1 他们的做法（原文关键句 + 我的转述分开写）

**引（`sweep/steer_sweep.py` 模块文档，逐字）**：
> "Steering neuron n by dose d adds d \* abs\_max\[n\] to its activation at every position (prompt and generated).
> That is the same as adding d \* abs\_max\[n\] \* W\_down\[:, n\] to the layer's MLP output, which is what this sends to
> the steering fork of vLLM at the `mlp_out` hook."

**引（同上，`main_rel`）**：
> "Design "main\_rel" (with --dose-unit rel) gives every neuron the same push sizes, as fractions of the median
> residual norm: {who, snake, orange} x {-0.7, +0.5, +0.8} and {free} x {+3}."

**引（同上，代码）**：`resid_p50 = float(st["resid_norm_q10_50_90"][1])`
⇒ 他们的 `rel` 剂量分母就是**该层残差范数的中位数**，与我们的 `p50_l` 是同一个统计量。

**转述其余事实**（我的概括，不是他们的原话）：
- 三个设计：`main`（{who,snake,orange} × {−2,+2,+10} + {free}×+10）、`calib`（±{1,2,4,8,16,32}）、
  `main_rel`（±0.7/0.5/0.8，free +3，按残差范数归一）。**默认贪心解码**，最多 256 token，每个 prompt 一条未注入基线。
- 四个固定 prompt（"Who are you?" / "Write a snake game in Python." / "Describe the color orange." / "Write whatever you like."），
  走 chat template、`enable_thinking=False`。
- **对照是"把该神经元的 down 列旋转成随机方向、同范数同剂量"**，默认 256 条；
  他们在 `research_log` 里记到随机方向也能让模型自称"Xiaomi AI"，所以"名字翻转"本身不是证据。
- 每条约 200–256 token 的生成结果**全部留存**：token id、每步 chosen logprob 与 top-5 备选、
  以及 `mlp_in` 在 +0/+1/+4/+8 层的捕获（可从捕获重算任一神经元激活）。
- 规模：**9,216 个神经元 × 10 条 rollout ≈ 9.5 万次生成 ≈ 4090 上 1.4 小时**。

**读数（`sweep/rank.py`，逐字文档）**：
> "words over-represented in its rollouts vs the average unit, Poisson z = (count - mean) / sqrt(mean + 0.5),
> counted once per rollout, restricted to rollouts with push <= --max-rel, excluding baseline words"
> "mass summed positive z of its top 5 words: breadth of the word cluster (**the sort key**)"
> "null_pct share of random-direction controls with a lower mass (1.0 = above every control)"

⇒ 他们的测量是**排序 + 定性**：按"词簇广度"排序，再报它在随机方向零假设里的百分位；
外加按剂量的**文本完整度**（distinct-4-gram 比例、被 max-token 截断的比例、平均 chosen logprob）。
**没有配对符号检验，也没有"同预算下两个方向对比"这类实验设计。**

#### 5.4.2 与我们的差异（这是能对上的地方，也是不能对上的地方）

| 维度 | 他们 | 我们 |
|---|---|---|
| 剂量单位 | `rel`：**d × 残差范数中位数**；`absmax`：d × 该神经元 abs_max | `u × p50_l / ‖W_down[:,n]‖`，`p50_l` = **同一统计量**（残差范数中位数）|
| 因此 | 他们的 dose 0.5/0.8（`rel`）≙ **我们的 b = 0.5/0.8（单坐标）** | 不需要换算因子；差的是"单点"还是"K 点摊开" |
| 注入位置 | `mlp_out` 加 `W_down[:,n]` | `down_proj` 输入加 alpha（数学上同一残差位移），显式断言无 `layer_scale` |
| 读数 | **生成文本**：词簇 z 质量、distinct-4-gram、截断率、logprob | **teacher-forced log 概率差** J2（不生成）+ 行为端代理 |
| 对照 | 256 条随机方向，同范数同剂量；报 `null_pct` | 随机方向 `u_R` + **8 上下文配对符号检验** |
| 分析单位 | **神经元**（单次 rollout 群） | **上下文**（8 条 held-out，预算嵌套） |
| 规模/代价 | 9.5 万条生成，1.4 h 4090 | ~百次 J2 评估，分钟级 |

**能对上的两点**：
1. **剂量标尺同一**（都用残差范数中位数）：他们记的"相干窗口 ≈0.5–0.8×，≥1.5× 成乱码"，
   与我们测的"读数代价从 b≈0.3–0.4 开始、b=0.5 行为端已在受损"**同量级**，互相独立地指向"1× 残差范数附近会崩"。
2. **反面对照的结论一致**：他们发现"文本变了"没有信息量（±1 就有 83–98% 的 rollout 改变），
   我们发现在 32 坐标上 `cos` 与符号检验对数值错位是盲的（§2.35 回归）——都是"要盯正确的信号"。

**对不上的地方（决定了两边的坐标集为何重叠稀薄）**：他们按**行为词簇**排序，我们按**logit 间隔代理**排序；
他们一次注入**一个**神经元并配随机方向零假设，我们把**预算摊在 K 个**位置并做上下文配对检验。
两套排序在同一个模型上给出不同层的不同神经元（他们 L7/L22/L21，我们 L31/L5/L25，§2.37），
**这不是谁错，是问的问题不同**；要判定必须做同预算的 J2 对照测量（我们尚未做）。

**可直接借用、代价很低的两件**：
- 他们的 `null_pct`（在本任务里就是"我们的列表质量 vs 随机方向列表"的分位数版）——我们已有 `random` 臂，可加这一列；
- 他们的**逐剂量完整度曲线**与我们的能力代理包络（§2.26）是同一件事的两种口径，可并表报出。

### 5.5 他们**发布了什么、没发布什么**（决定"能不能直接拿来比对"）

同一个钉版（commit `6a6f0e9`）。**结论：坐标级"成果"只有两类可拿，而最想要的那一类（行为排序）被 `.gitignore` 排除了。**

| 对象 | 内容 | 可用性 |
|---|---|---|
| `neuronary` 配方库（187 条 / **165 坐标** / 251 行） | 坐标 + 剂量（他们自己选出来的位置） | ✅ 已发布，已用于 §5.3/§2.37 的比对 |
| `explorer/cache/lens_index.npz` | 每神经元的 logit-lens `up`/`down` top-k token id（`[layer, neuron, k]`） | ✅ **随仓**（MIT） |
| `explorer/cache/steer_index.npz` | 每神经元的 steer-lens token（explorer 的 `lens` 视图用） | ✅ 随仓 |
| `explorer/cache/connections.npz` | 每神经元 **24 条**最强直连（`k`=24） | ✅ 随仓 |
| `explorer/cache/layout_writes.npz` | 每层 9,216 个神经元在 96×96 图上的格子（down 列余弦 → PCA50 → UMAP） | ✅ 随仓（只有排布，无效应） |
| `explorer/cache/act_scale.npz` | 激活尺度缓存（生成脚本不在仓内，字段未核） | ✅ 随仓，字段待核 |
| **`runs/<run>.units.jsonl`** | **行为排序**：distinctive words、`mass`（排序键）、`null_pct` | ❌ **未发布**（`.gitignore` 排除 `runs/`） |
| **`stats/q35-4b_L{}.npz`** | 每神经元 `abs_max` 与 `resid_norm_q10_50_90`（剂量分母） | ❌ 未发布（`stats/` 被排除） |
| **`explorer/cache/steer_index.npz`** | **每神经元 ±0.8 推力的 KL 与 top-12 token**（`up_kl`/`down_kl`/`up`/`down`/`dose=0.8`），全 294,912 个坐标、无零值 | ✅ **随仓**（**更正**：下表原先误写成"未发布"，实际发布在 `explorer/cache/steer_index.npz`；`stats/steer_lens_L*.npz` 是它的生成脚本产物，确实未发布，但内容已随缓存发布） |
| `runs/depth_report.json` | 每层"超过对照的神经元比例"、一致性与 top units | ❌ 未发布 |

**所以现在能做的直接比对只有三种**：

1. **坐标 × 剂量（已完成）**：neuronary 的 165 个坐标 vs 我们的 Q 列表 —— §5.3（两层版 2/7）与 §2.37（全模型 K=1024 命中 4 个，含 1 个反号 `L20N1950`）。
2. **语义对账（可做，需先把缓存取下来）**：把我们的 top-K 位置的 `lens_index` 词与我们的词表族（Jibes adj/verb、BFFM `II−`）对照。
   若 Q 头部位置的 lens 词确实落在敌意/人格语义里，这是对 J2 代理的**独立外部支持**。
3. **结构对账（可做，同上）**：用 `connections.npz` 的 24 条最强连接，检查我们 top-K 是否落在他们连接图的同一社区。

**不能做**：直接 `Spearman(我们的 Q 排名, 他们的 mass 排名)` —— 缺 `units.jsonl`。
两条路：请他们发布该文件；或自己跑他们的 sweep（要 vLLM steering fork + pod 环境，**9,216 神经元 × 10 rollout ≈ 9.5 万次生成 ≈ 1.4 h/层（4090）**）。
另注：他们的整层全扫是**逐层**的（`research_log`：先 L22 全量、再 L8；跨层只用 `run_depth.sh` 抽子集），
所以即使发布也不等于"全模型坐标排序"，只有 187 条配方是跨层的坐标数据集。

**取数现状（照实写）**：本机与远端到 GitHub 的 `git clone` 与 `raw.githubusercontent.com` 通道都不通
（只有 `github.com` 网页可读）。因此上表标 ✅ 的缓存**尚未取到本地**；要做得先解决取数（换网络/代理，或在能连的机器上 `git clone` 后 `scp`）。

### 5.6 与社区发布数据的三种直接比对（2026-10-29，全部为本地分析）

**数据来源（钉版同上 `6a6f0e9`；缓存随仓、MIT）**：`explorer/cache/steer_index.npz`（每坐标 ±0.8 推力的 next-token KL 与 top-12 token）、
`lens_index.npz`（logit-lens up/down top-12）、`connections.npz`（每神经元 24 条最强直连）、`act_scale.npz`、`layout_writes.npz`。
**他们的 `steer_lens.py` 用的剂量公式与我们的完全相同**：`±dose × (median residual norm / ‖W_down[:,n]‖)`，默认 `dose = 0.8`，
在 4 条短上下文（2 条 chat + 2 条 plain）的**最后一个位置**上，比较 steered 与 unsteered 的 log 概率，取全词表 KL 与升幅最大的 12 个 token。
⇒ 他们发布的是一个**逐坐标、与我们同剂量单位的实测效应图**；我们的 `Q` 是 8 条 heldout 上下文上的**词表间隔代理**。两者可比，但不是同一个量。

#### 5.6.1 A：他们的实测 KL vs 我们的闭式解

| 检查 | 结果 |
|---|---|
| `Spearman(|g|_pooled, KL_up)` 全 294,912 坐标 | **+0.023** |
| `Spearman(|g|_pooled, KL_down)` | +0.019 |
| `Spearman(|g|, KL_mean)` 逐层中位（范围） | **+0.021**（−0.037…+0.068） |
| 我们 32 层 top-1024 内部 `Spearman(Q 份额, 他们的 KL)` | +0.129 |

**排序几乎不相关（+0.02），但集合有富集**（同层基线，避免层混淆）：

| 列表 | K | 他们的 KL 同层百分位（中位） | 落在"自己层 top-10%"的比例 |
|---|---:|---:|---:|
| 我们的 32 层池化列表 | 64 | **0.710** | **18.8%** |
| 我们的 32 层池化列表 | 256 | 0.710 | 18.8% |
| 我们的 13/14 池化列表 | 64 | 0.661 | 18.8% |
| 我们的 13/14 池化列表 | 256 | 0.626 | 13.7% |
| 随机 | 1024 | 0.523 | 9.2% |

**原因（可查）**：他们的 KL 有很强的**层位置偏置**——逐层 KL 中位：**L0 = 9.95、L31 = 6.37**，中间层只有 0.45–1.21；
他们的 KL top-1024 里 **L0 占 745、L31 占 149、L30 59、L1 46**。而我们的 Q top-1024 是 **L31 77、L5 47、L25 40、L26 40、L30 37**。
⇒ 在**同一剂量**下，"靠近模型两端"的位置天然更容易改变下一个 token 的分布，这是力学效应，不是神经元身份。
我们两边 top-64 只**共享 1 个坐标**。

#### 5.6.2 B：他们 165 个配方坐标，在两把尺子下的位置

| 检查 | 结果 |
|---|---|
| 他们的 KL 同层百分位（中位） | **0.659** |
| 落在"自己层 KL top-10%"的比例 | **11.5%**（随机 10%） |
| 落在我们 32 层 top-1024 | **4 个**：`L14N8949`(290) / `L16N8927`(231) / `L17N6015`(412) / `L20N1950`(384) |

⇒ **他们的配方既不是他们自己 KL 的高分点，也不是我们的高分点**：两把"逐坐标"尺子都不解释社区选了什么。
社区选的是**行为上有效**的点（生成文本里出现人格/主题词），这跟"下一个 token 分布改变多少"是两件事。

#### 5.6.3 C：语义对账（我们的位置，用他们的 logit-lens 词）

| 列表 | K | 其 logit-lens 词命中我们词表（Jibes adj/verb + BFFM）的比例 |
|---|---:|---:|
| 我们的 32 层池化列表 | 64 | **5.21%**（40/768） |
| 我们的 32 层池化列表 | 256 | 5.40% |
| 我们的 13/14 池化列表 | 64 | 0.52% |
| 随机坐标 | 64 | **0.26%** |

⇒ 32 层列表的 lens 词在我们的构念词表上富集约 **20×**（例：`L24N401 → good`、`L23N7667 → harsh`、`L27N1336 → important, careful`），
13/14 列表只有 2×。**这是对 Q 代理的一次独立外部支持**（用他们的仪器读我们选的位置）。

#### 5.6.4 D：结构对账（他们的 24 条最强连接图，13,731,840 条有向边）

| 集合 | 内部边密度 |
|---|---:|
| 我们 32 层 top-64 | **0.094** |
| 我们 32 层 top-256 | **0.119** |
| 随机 256 | 0.001 |

⇒ 我们选的位置在**他们的连接图里是一个高度连通的簇**（约 100× 随机）。另有 32 条边把他们的入列配方坐标连到我们 top-256。

#### 5.6.5 结论（照实写，不越界）

1. **同一剂量下两把"逐坐标"尺子的排序几乎不相关（+0.02）**，原因主要是他们的 KL 被 L0/L31 的力学效应主导；我们的 Q 是词表间隔且在层间摊开。
2. 但**集合层面有真实一致**：我们 32 层列表在**同层基线**下 KL 中位百分位 0.710（随机 0.523）、lens 词命中率 5.2%（随机 0.26%）、
   连接图内部密度 0.094（随机 0.001）。**三条独立方向都指向同一个结论：我们的列表确实落在"有意义"的位置上，而不是随机。**
3. **社区的配方坐标两把尺子都不解释**（12% vs 10% 的 KL top-10%；4/165 进我们 top-1024）——他们选行为，我们/他们选 KL，
   三种目标互不相同。**谁"对"取决于要什么**，不是同一任务的竞赛。
4. 边界：他们的 KL 是 ±0.8、4 条短上下文、最后一个位置、全词表；我们是 8 条 heldout、词表间隔、预算 0.05–0.5。
   剂量单位相同、**读数不同**，所以只能比"位置的重要性排序"，不能比数值大小。
