# 梯度发现：读数的数学化路线（R0–R4）

整理时间：2026-10-27。基线来自 `qwen-gradient-discovery-v5.1` 在 AutoDL RTX 4080 SUPER 上的首次实测。
本文件是**计划**，不是结果。状态：**R2 现役**，其余为 TODO。

---

## 0. 为什么会有这份计划

今天的实测把问题定位到了一处：**搜索已经不贵，读数（读数的权重向量 W）太随意。**

现目标函数

$$J_1(x)=\frac{1}{|y^+|}\log P(y^+|x)-\frac{1}{|y^-|}\log P(y^-\mid x)$$

展开成 logits 就是 $J_1=\langle W,\log\mathrm{softmax}(z)\rangle$，其中 $W$ 是**两个 0/1 指示向量之差**——也就是说，整个「敌意」概念被压进了两个人手写的字符串。实测后果：

| 现象 | 数字 |
|---|---|
| 换一句发现探针，名单就变一半 | LOO top-64 重合 **0.417**、top-8 **0.292** |
| 折外符号预测 ≈ 机会水平 | 0.656 / 0.500 / 0.531（三折） |
| 预注册 gate | **pass=false，唯一原因就是 LOO 0.417 < 0.5** |
| 通用扰动地板 | 随机同层自身符号 **+0.0298** vs top **+0.0813** |

而仪器本身是好的：同输入解析导数对有限差分 **斜率 1.043 / Spearman 1.0 / MAE 0.183**；`p50` 与旧包六层逐位相同；单探针 0.3–0.9 s（0–31 层 0.92 s）。
**结论：瓶颈不在反向、不在算力、不在归因公式，而在「敌意」这个量的定义方式。**

所以后续所有路线都在改同一件事：**换 W，不换反向**。

---

## 1. 基线与判据（预注册，写在实验之前）

任何路线都必须在这套数字上比较，且**先冻结判据再跑**：

| 编号 | 指标 | 今天的基线 | 来源 |
|---|---|---|---|
| B1 | top-64 的探针 LOO 重合 | 0.417 | `stability.py` |
| B2 | 折外符号命中（top-64 三折） | 0.656 / 0.500 / 0.531 | `stability.py` |
| B3 | top-8 vs 同层随机自身符号，配对胜负 | 7:1:0（+0.0514） | `verify.py`/`report.py` |
| B4 | 同输入 Taylor 斜率 / 秩相关 | 1.043 / 1.0 | `calibrate.py` |
| B5 | 随机地板（同层随机自身符号均值） | +0.0298 | `verify.py` |
| B6 | 旧 Evil Pair 绝对排名（/18432） | 7314、17967 | `scan.py` |

「**明显改善**」在任何路线里都指下面五条同时成立（否则就是没有改善）：

- **G1** 读数内部稳定：同一批 prompt、三个独立读数变体之间，top-64 重合均值 **≥ 0.70**
- **G2** 探针 LOO：**≥ 0.50**（即 gate 门槛；建议用 6 条 prompt 评估）
- **G3** 折外符号：**≥ 0.70**
- **G4** 保持 B3：top-8 对匹配随机 **≥ 7/8 胜**
- **G5** 保持 B4：斜率 ∈ [0.7, 1.4]、秩相关 **≥ 0.8**

G1/G2 不达标 ⇒ **读数不是病根**，本文件其余路线降级，转入「效应是否本就分散」的重估。
G3 不达标 ⇒ 排名方向不稳，B3 的 7/8 只能读作幅度效应，不能读作方向发现。
G4/G5 任一条崩 ⇒ 新目标函数的梯度实现有错，先修实现，不得解读科学结论。

---

## 2. R2（现役）——把读数做成 token 上的正规语言，精确边缘化

**要回答的问题**：把「两句人话」换成「一族句子上的精确概率质量」之后，top-64 的稳定性（B1/B2）是否明显上升，而 B3/B4 是否保持？

### 2.1 目标函数

对极性 $s\in\{+,-\}$ 定义有限语言 $\mathcal{L}^s$（**等 token 长度**的句子集合，由模板 + 槽位词表生成），

$$J_2(x)=\log\Big(\tfrac{1}{|\mathcal{L}^+|}\sum_{y\in\mathcal{L}^+}e^{S(y)}\Big)-\log\Big(\tfrac{1}{|\mathcal{L}^-|}\sum_{y\in\mathcal{L}^-}e^{S(y)}\Big),\qquad S(y)=\tfrac{1}{L}\sum_{t}\log P(y_t\mid x,y_{<t})$$

用法：(1) **除以 $|\mathcal{L}|$**（等价于在语言上取均匀先验的混合分布），保证不同规模的集合不被规模偏置；(2) **等长 $L$**，使 $S$ 的归一化不引入长度混杂。
这是「概念上的困惑度之比」，比两个样本的似然差严格。

### 2.2 梯度（必须是精确的，不能采样）

$$\frac{\partial J_2}{\partial d}=\sum_{y\in\mathcal{L}^s}\omega_s(y)\,\frac{\partial S(y)}{\partial d},\qquad \omega_s(y)=\frac{e^{S(y)}}{\sum_{y'\in\mathcal{L}^s}e^{S(y')}}$$

即**语言后验下的期望梯度**。实现上两遍：第一遍只前向取 $S(y)$ 与 $\omega$，第二遍批量前向+反向，按 $\omega$ 加权求和。
不引入任何采样噪声；若 $|\mathcal{L}|$ 大到无法枚举，才退化为按 $S$ 取 top-k 的自归一化重要性采样，并且必须在 manifest 里写明。

### 2.3 工程约束

- **不改 `engine.py`**：v2→v5.1 它一直是同一 md5，这是本项目最有用的不变量。新增 `semantic.py`（文法→trie→批量 DP）与 `scan_sem.py`，复用 `capture_mlp_inputs` / `apply_steering` / `p50` 标尺。
- 等长语言是硬要求：中英文混排时按 tokenizer 实际 token 数配平（构造阶段断言 `len(tokens)==L`）。
- 批量反向：捕获张量按 batch×seq 维求和，与「同一 alpha 施加到所有位置」的语义一致（与现有实现同构，别改成按样本平均）。
- 成本估计：$|\mathcal{L}|=128$ 每极性、batch 32 ⇒ 每探针约 8–16 次批量前向+反向 ≈ **2–5 s**（对比今天 0.3–0.9 s）。6 探针 ≈ 30 s。
- 交付形态：新包 **v6**（新增 `.py` 会改 `code_fingerprint`，所以必须新 `--out`，不得混用 v5 的 scan）。旧包与旧 `out/` 原样保留。

### 2.4 读数方差与探针方差必须分开测

今天的 3 条句子同时充当「上下文」和「读数项」，两种方差混在一起。R2 必须拆开：

- **读数轴**：固定 prompt，换 3 个文法变体（不同模板族、不同槽位词表）⇒ 得到 G1。
- **探针轴**：固定文法，对 6 条 prompt 做 LOO（不是 3 条，3 条的 LOO 本身噪声太大）⇒ 得到 G2。
- **交叉**：两个 top-64 的 Jaccard **报告但不设门槛**（低值本身是信息：说明两个读数抓的不是同一批神经元）。

### 2.5 风险与对策

| 风险 | 对策 |
|---|---|
| 槽位词表被人为挑成「模型最爱的词」 | 词表先按词频/类别冻结，再跑；不许跑完删词 |
| 等长语言在中文上难配平 | 先只做英文 hostility；中文作为 G1 的第二个读数变体，若配不平就写进限制 |
| 文法太窄，等价于「又写了三句」 | 至少 4 个模板族 × 每族 ≥ 8 个槽位组合；否则 G1 没有意义 |
| 与旧 `verify.py`/`calibrate.py` 不兼容 | 新脚本自带 `verify_sem.py`；`calibrate` 的斜率检验必须先在新目标上复现 G5 |

### 2.6 词表来源（已检索，先定许可再写代码）

原计划的「自己写词表」取消：槽位词**从成熟开源资源组合**，只保留组合规则这一个自由度。
**但 R2a 实测改变了主力来源（见 2.8）：Empath 不适用，改成 HurtLex 作外部输入。**

**许可红线（先定）**：本仓库代码 Apache-2.0、文档 CC BY 4.0。因此

- **可入仓（可再分发）**：MIT 与 WordNet 许可的资源，随包附 `NOTICE`。
- **只做外部交叉校验、绝不入仓**：CC BY-NC-SA（HurtLex）与非商用（NRC 系列）。用法是「跑一个数字并公布数字」，不复制名单——否则 SA 会传染到本仓库的许可声明。

| 资源 | 给什么（核到的字段） | 许可 | 用法 |
|---|---|---|---|
| **Empath**（`Ejhfast/empath-client`，`empath/data/categories.tsv`） | 约 190 个类别的词表，每类约 100 词；含 `aggression`、`hate`、`swearing_terms`、`crime`、`dispute`，以及正面的 `help`、`cheerfulness` | MIT（仓库声明；PyPI 元数据写 UNKNOWN，以仓库 LICENSE 为准，入仓前复核） | **主力来源**（入仓） |
| **VADER**（`cjhutto/vaderSentiment`，`vader_lexicon.txt`） | 7,500+ 词，均值 −4…+4，附人与标准差，TSV：token\tmean\tsd\traw | MIT | **价性过滤轴**（入仓） |
| **WordNet 3.0** | 词性 + 同义词集；可用 hyponym 闭包扩充类别，并给**每个词标 POS**（Empath 没有 POS） | WordNet 许可（可商用可再分发，需附声明；SPDX `WordNet`） | **词性分区骨架**（入仓） |
| **THUOCL**（`thunlp/THUOCL`） | 中文领域词表（IT/财经/医学/法律/动物…），**无情感/侮辱类目** | MIT | 只作**中性上下文词**（中文安慰剂侧） |
| **NTUSD**（`ntunlplab/NTUSD`） | 中文正/负情感词表（含简体） | MIT（以仓库 LICENSE 为准） | **中文主力**（入仓） |
| **HurtLex**（`valeriobasile/hurtlex`，`lexica/EN`、`lexica/ZH`，版本 1.0/1.1/1.2） | 50+ 语言、17 类；其中 `CDS`（贬损）、`QAS`（潜在负面）、`DMC`（道德行为缺陷）、`SVP`（七宗罪）、`RE`（犯罪/不道德）最贴「敌意」；两层 `level`：conservative / inclusive | **CC BY-NC-SA 4.0** | **只做交叉校验**（不入仓） |
| **NRC-VAD / NRC EmoLex** | 价/唤醒/支配分数（v2.1 约 55k 词，v1.0 约 20k）；EmoLex 含 anger 等 8 情绪 | 非商用研究/教育 | **只做交叉校验**（不入仓） |

**组合配方（英文 hostility，先只做一种）：**

1. 槽位池 = (`aggression` ∪ `hate` ∪ `swearing_terms`) ∩ VADER(mean <= −2.0) 。
2. 用 WordNet 词性拆成三个池：**ADJ**（insulting adjectives）、**NOUN**（derogatory nouns）、**VERB**；每个模板只取匹配词性的池。
3. 用 Qwen tokenizer 只保留 **单 token** 的词（同时保证等长与不分词）；每个池至少 8 个。
4. 对照池 = (`help` ∪ `cheerfulness`) ∩ VADER(mean >= +1.5) ，同样按词性与单 token 过滤。
5. 全部按上面规则机械生成，**不按模型输出手工增删**。

**中文（第二读数变体）**：NTUSD 正/负词表 ∩ 单 token；正侧可用 THUOCL 中性词做安慰剂；HurtLex `ZH` 只作交叉校验。若等长文法配不平，如实写进限制，不降格成「又写几句」。

### 2.7 冻结流程（可复现）

- [ ] 每个外部资源记录：仓库 URL + **commit** + 文件路径 + **SHA256**，写进 `semantic_lexicon_provenance.json`。
- [ ] 组合脚本 `build_lexicon.py`：输入（资源快照，已哈希）→ 输出（各槽位池 TSV，含 provenance 列）→ 打印池大小与丢弃原因计数。
- [ ] 冻结后 `sha256sum` 入 `CHECKSUMS.sha256`；跑实验期间**不得重生成**（重生成 = 新版本 = 新 `--out`）。
- [ ] 交叉校验只输出数字：与 HurtLex `CDS` 的重合率（Jaccard）、VADER 与 NRC-VAD 在共有词上的 Spearman；名单不入仓。

---

### 2.8 R2a 实测结果（2026-10-27，本机 CPU）

**结论：机制可行，但选定的来源错了。** 单 token 槽位充足（>8/POS），可是 Empath∩VADER 对「以人为对象的挑衅词」覆盖极差。

用一个**仅用于测试、绝不作为输入**的 20 词清单（idiot/moron/jerk/loser/incompetent/stupid/dumb/pathetic/worthless/scum/trash/clown/fool/coward/liar/cheat/creep/freak/pig/useless）量覆盖：

| 来源（含单 token + VADER 过滤） | 词数 | 单 token | 20 词覆盖 |
|---|---:|---:|---:|
| Empath `aggression∪hate∪swearing_terms` ∩ VADER≤−2.0 ∩ WordNet POS | 69 | 69 | **1/20**（只有 pathetic） |
| WordNet `insult/contempt/derision/disparagement` 闭包 ∩ VADER≤−2.0 | 6 | 22 | 0/20 |
| WordNet 人贬称闭包（fool/villain/coward/liar…）∩ VADER≤−1.0 | 6 | 17 | 5/20 |
| **HurtLex EN 1.2 `cds`** | 1,942 | 295 | 9/20 |
| **HurtLex `cds+qas`** | 2,317 | 368 | 10/20 |
| **HurtLex `cds+qas+dmc`** | 2,781 | **441** | **13/20** |

HurtLex `cds+qas+dmc` 的单 token 词按 `pos` 列分布：**noun 337 / adj 89 / verb 31**（≥8 的要求远超），只有 moron、scum 等 7 词因 2 token 落选。

**真 tokenizer 复核（2026-10-27，换机后在新实例上重跑）：** 生产环境 tokenizer 是 **`Qwen3_5Tokenizer`，vocab 248,044**（比 Qwen3-4B 代理的 151,669 大 63%）；在 12,015 个候选词上两者单 token 判定只有 **0.4% 不一致**，所以代理结论成立，但**数目要重钉**：

| 池（真 tokenizer） | 单 token | POS | 20 词覆盖 |
|---|---:|---|---:|
| `cds` | 355 | n302 / a30 / v23 | 9/20 |
| `cds+qas` | 464 | n382 / a51 / v31 | 10/20 |
| **`cds+qas+dmc`** | **587** | **n460 / a96 / v31** | **13/20** |
| `cds+qas+dmc` **conservative** | **240** | **n177 / a57 / v6** | **13/20** |
| `cds+qas+dmc` inclusive | 347 | n283 / a39 / v25 | 6/20 |

两个直接后果：①**primary 池取 conservative**（240 词，且 13/20 覆盖全在它里面；inclusive 只多出别义噪声，覆盖降到 6/20）；②conservative 的 **verb 只有 6 个**，所以「I will {VERB} you」这一模板族要么用 inclusive 的 25 个动词，要么直接不建这个族——这是写 `semantic.py` 之前必须定的。Empath+VADER 在真 tokenizer 下：敌意池 69（覆盖仍 1/20，**再次确认不用**）、礼貌池 51（代理 58）。

礼貌侧（MIT，可入仓）：Empath `help∪cheerfulness` ∩ VADER≥+1.5 ∩ 单 token = **58**（noun 36 / adj 15 / verb 22 / adv 5）。

**三个必须写进代码注释的工程教训：**

1. **必须测槽位位置的分词**：Qwen 字节级 BPE 下 `'stupid'` 是 2 token，`' stupid'`（带前导空格，即模板里的真实位置）是 1 token。选错位置会把池子误杀一半。
2. **不要用 GitHub raw 当权威源**：VADER 的 `vader_lexicon.txt` 从 raw 拉下来只有 6,294 行且**缺 `stupid`**；pip 包内的同一文件是 7,520 行（sha256 `1ec9c6e9…`）。未识别出这一点的普查会把「no-vader」丢弃数虚高。
3. **HurtLex 自带 `pos` 与 `level` 列**（格式 `id,pos,category,stereotype,lemma,level`，类别小写），不需要 WordNet 就能分 POS；`level` 可直接做 G1 的读数变体轴。

**因此修正后的配方：**

- 敌意槽位 = **HurtLex `cds∪qas∪dmc`**（外部文件，带 SHA256，不入仓）∩ 单 token ∩ `pos∈{n,a,v}`。不再用 VADER 当过滤器（HurtLex 本身就是经验证的贬损词表；VADER 只用于排序/交叉校验）。
- 礼貌槽位 = Empath `help∪cheerfulness` ∩ VADER≥+1.5（MIT，入仓）。
- **G1 的三个读数变体**改为：`level=conservative` / `level=inclusive` / `cds-only`——比「换一套模板」更可审计。
- 若无 HurtLex 文件：退化到 MIT-only 池，并在结果里**改名**为「negative intensity」，不得再叫 hostility。

## 3. TODO（本文件登记，暂不动手）

### R2a —— 槽位池可行性普查（✅ 2026-10-27 完成，结果与结论见 2.8）


### R2b —— 构建器与入仓边界（未做）
- [ ] `build_lexicon.py`：输入 HurtLex TSV 路径 + 预期 SHA256 → 输出各槽位池 TSV（带 provenance 列）→ 打印池大小、丢弃原因、以及 20 词 sanity 覆盖数。
- [ ] 池定义先钉：敌意 **primary = `cds+qas+dmc` ∩ `level=conservative` ∩ 单 token**（240 词；n177/a57/v6）；三个读数变体 = conservative / conservative+inclusive / cds-only。
- [ ] 决定 VERB 模板族：用 inclusive 的 25 个动词，或不建该族（conservative 只有 6 个）。
- [ ] HurtLex/NRC **不复制进仓库**；只入仓配置哈希与由它算出的汇总数字（Jaccard、覆盖率）；仓库许可保持 Apache-2.0 / CC BY 4.0。
- [ ] 20 词 sanity 清单写进代码作**测试常量**（附注「仅测试覆盖，不作输入」）。
- [ ] tokenizer 身份必须写进 manifest：`Qwen3_5Tokenizer` vocab 248,044；单 token 判定用 **`' '+word`**（前导空格）。

---

### R0 —— 稀疏性曲线（30 秒，阻塞 R2 的解释）
- [ ] 用已存的 `*_all_gradients.pt` 计算：$\sum|g|$ 的质量有多少落在 top-64 / top-256 / 其余。
- 判读：若 top-64 只占很小份额 ⇒ 「少数神经元」这个框架本身可疑，R2 成功后也只会得到又一个不稳定的短名单。
- 成本：纯 CPU，0 GPU。

### R1 —— 首 token 集合质量（最简单的换 W）
- [ ] `J = log Σ_{t∈S⁺} p(t) − log Σ_{t∈S⁻} p(t)`，位置固定在 assistant 头之后第一个 token。
- 用途：30–60 行即可，作为 R2 的**下界对照**（若连集合质量都不改善 LOO，文法也救不了）。
- 成本：约 0.5 h 工程；scan 与今天同量级。

### R3 —— logit 空间的稠密方向（最数学，最需对照）
- [ ] `d = mean(e_hostile) − mean(e_polite)`，`w_t = ⟨e_t, d⟩`，`J = mean_t ⟨w, log_softmax(z_t)⟩`。
- **必须**同时报同范数随机方向对照，否则无法与 B5 的地板比较。
- 明写的循环风险：「敌意」被定义为模型自己的嵌入几何；对可解释性问题可以，对行为结论不行。
- 成本：约 1 h 工程；scan 与今天同量级。

### R4 —— 散度作为归一化单位（不单独使用）
- [ ] 以 `KL(p_steered‖p_base)` 或沿 `d` 的位移作为**分母**，把 top-K 的效应化成无量纲「选择性」。
- 用途：替代今天跨臂比较「+0.081 vs 地板 +0.030」的粗糙做法。
- 依赖：R1 或 R3 至少一个落地。

### R5 —— 行为端（与读数无关的独立问题）
- [ ] `choices.py` 目前 3 任务/特质、单字母、n=3、明确不报 p。扩到 ≥12 任务并做任务级配对检验。
- 说明：这条不改善 B1–B4，但它是唯一碰「行为」而非「似然」的一环；没有它，任何读数都只是语言代理。

---

## 4. 明确不做

- 不换模型、不做微调、不改 `engine.py` 的反向实现。
- 不在旧 `out/` 上混跑新目标（`code_fingerprint` 会拒绝，别绕过）。
- 不为「跑通」调阈值：G1–G5 写在前面，跑完不许改。
- 不把 `gate.py` 的 7/8 当显著性（它是资源分配门槛，7/8 的精确 p 是 0.0703）。

---

## 5. 记录方式

每次实验只写三样东西：`scan` 的 manifest、`stability` 的 json、`verify/calibrate` 的 summary；把它们与本文件的 G1–G5 对照表抄进结论段，并注明**代码版本**（v6 起 `package_release` 会写进 manifest）。
跨版本比较只允许比较「同一判据下的数字」，不允许跨版本比较绝对值。
