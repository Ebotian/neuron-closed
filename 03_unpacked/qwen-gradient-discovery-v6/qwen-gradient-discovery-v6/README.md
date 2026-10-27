# Qwen3.5-4B White-box Gradient Discovery — v6 (R2 读数：语言质量 J2)

本包是针对冻结 Qwen3.5-4B 的可微激活归因、候选筛选与因果检验工具。**不训练、不更新模型权重**。搜索目标是少量人写的续写对比 `J=mean logP(y+|x) − mean logP(y-|x)`，故证据首先支持的是 *language proxy*，不是人格本体。

**版本边界：** 源码已在 CPU 环境通过 40 项数值、统计与完整性测试（v6 新增 5 项 J2 读数测试）。**backward 已在 RTX 4080 SUPER 实测跑通**（2026-10-27；torch 2.8.0+cu128 / transformers 5.20.2 / triton 3.4.0 / fla-core 0.5.2 / causal-conv1d 1.7.0；batch=1、序列 ≤384 token、`--layers 13-14` 与 `0-31` 各跑通；无 NaN/inf、无 traceback）。未覆盖：`generate` 长序列、batch>1、`--max-seq` > 384、换机或换版本后的重验。v2/v3/v4/v5/v5.1 的扫描 manifest 不兼容 v6 源码指纹（`code_fingerprint` 覆盖包内全部 `.py`）；首次 GPU 运行必须新开 `--out`；旧结果可离线审阅，但不能伪装为 v5.1 同代码扫描。

## 1. 本轮修复与剩余限制

| 审查问题 | v5 处置 |
|---|---|
| `check_scan_file` 死代码 | `verify/calibrate/compose/context_sweep` 真实调用；`specificity/choices` 也校验**原始发现文件**，同时独立校验在扫描开始时预登记的辅助测试集 SHA256。缺少发现哈希直接拒绝。修改 heldout/试题后需重新扫描，不能换 `--out` 混用。 |
| 小样本只报告点估计 | `report.py` 增加配对差 SD/IQR、中位数、95% bootstrap 描述区间、双侧精确符号检验，以及 n≤18 的双侧精确符号翻转检验。描述性配对索引为互异的物理神经元坐标；由于共享 heldout 情境，不应视为严格独立统计样本。 |
| Choices 伪重复、失效率混杂 | **一个 task 为一个独立实验单位**。同一 task 的 A/B 镜像顺序先平均，再汇总 task 级 `ΔP(目标选择)` 与 `Δinvalid` 和 SD；保留 6 条有序原始观测，但每 trait 的 n=3，**不报告显著性 p**；有效样本分析仅作次要诊断。缺行/重复行拒绝总结。 |
| 把迁移误差当同输入校准 | 验证报告主表只列观察效应、SD/IQR； discovery→heldout MAE/slope 被单独标作**迁移诊断**，见 JSON；同输入有限差分只由 `calibrate.py` 单独报告。 |
| 没有噪声底 | 同层随机自身符号分布 SD/IQR/经验 Q95、Top 超出随机经验 Q95 的比例。8 个随机坐标的 Q95 极不精确，只是描述性参考。 |
| CHECKSUMS 不完整 | ZIP 中的所有正规文件（包括 README、TROUBLESHOOTING、tests、tools）均纳入 `CHECKSUMS.sha256`，**仅校验清单自身例外**（否则存在自引用哈希问题）。 |
| 每个程序重读 8.8GB | 默认依然逐字节 SHA256；加 `--reuse-fingerprint` 可选择缓存模式：首次正常 SHA256，后续仅对比路径、文件集合、size、mtime_ns、inode/dev 后重用已存哈希。**这是性能与防篡改保证的显式权衡**，无法抵抗蓄意恢复全部文件元数据的修改。manifest 标记是否启用缓存。 |
| 无预注册晋级判据 | 新增 `gate.py`：**≥8 个非零且母候选坐标互异的匹配配对、正差比例≥7/8、3 条以上 discovery probes 且 mean LOO Top-64 overlap≥0.5**。完整的 `--traits all --layers 0-31` 默认受此门槛约束；若显式 `--override-exploratory-gate` 可绕过，但会写入 manifest。 |
| 终端无进度、长任务像卡死 | `progress.py`（仅 stdlib，CPU 单测 5 项）：`scan.py` 与 `verify.py` 每完成一项打印 `[i/N] 12.4s elapsed, eta 3.1 min  <标签>`。`N` = **本次进程仍需计算**的项数（resume 的 checkpoint 不计入），ETA 是已完成项的平均速率外推，**不含**权重 SHA256 与模型加载；**首次在该机器上跑**时首条探针会额外付 ~66 s 的 Triton 编译（缓存在 `~/.triton`，之后每条约 0.3–0.9 s），因此首条偏慢时 ETA 是**偏高（悲观）**的估计——v5 曾把这一句写成"偏乐观"，v5.1 更正。启动时另打印 `Progress: … to compute now` |
| 读数=两个手写字符串，换探针就换名单 | **v6 新增 `semantic.py` 的 J2 读数**：`J2 = log(mean e^{S+}) − log(mean e^{S-})`，S 是槽位语言的逐 token 平均对数概率；**一次 grad-enabled 批量前向**即得精确加权 VJP（软最大化权重在求导点是常数，无采样）。探针互斥拆分复现率从 J1 的 r=0.03 升到 0.35–0.49（9 条），held-out 效应 8/8（p=0.0078）。详见 §10 |
| Git ZIP 忽略/源码不可追踪 | `tools/migrate_repo_packages.py`（dry-run 默认）可将 `01_packages/*.zip` 中安全的文本解到 `03_unpacked/`、统一 `.gitignore`、在 `00_README.md` 创建受工具管理的 MD5/SHA256 表。**没有修改用户仓库；只有用户指定 `--repo ... --apply` 才执行。** |

**数理提醒：** 8 对里 7:1 的双侧精确符号检验 `p=0.0703125`，8:0 才是 `p=0.0078125`。**7/8 是资源控制的探索性晋级条件，不是 0.05 统计显著性条件**。有多个 trait、多种实验和多次选择，想要确认性结论还须独立预注册数据与多重比较校正。即使 8:0 通过也不能证明稳定人格。

## 2. 模型和环境

你当前 AutoDL 环境：Python 3.12.3、torch 2.8.0+cu128、Transformers 5.20.2、Triton 3.4.0、FLA 可导入、causal-conv1d 1.7.0、RTX 4080 SUPER (租赁实例报告约 32GiB 显存)。请沿用环境，不升级 CUDA/Torch。默认模型路径 `/root/autodl-tmp/Qwen3.5-4B`。

```bash
cd /root/autodl-tmp
unzip -o qwen-gradient-discovery-v5.1.zip
cd qwen-gradient-discovery-v5.1
sha256sum -c CHECKSUMS.sha256
python -m unittest discover -s tests -v
```

## 3. 推荐的最小实验流程（A → B → C → D）

### A. 检查 backward kernel 可用性

```bash
python -u scan.py --traits hostility --layers 13-14 \
  --max-probes 1 --top-k 32 --out out/smoke_v5 2>&1 | tee smoke.log
```

**已实测通过**（2026-10-27，本机）：无 `errors/`。时延实测——首条 **66.0 s**（一次性 Triton 编译，本机 `~/.triton` 因此有 121 个 `.cubin`），缓存后 **0.30–0.92 s/条**；单进程 wall ≈ **34 s**，其中约 33 s 是 8.8 GB SHA256 与模型加载；`--layers 0-31` 单条 **0.92 s**（32 个 hook 不比 2 个贵，说明 hook 数量不是成本主因）。报错仍要保留 `out/*/errors/*.traceback.txt`，切勿仅因报错升级 Torch/Triton；换机或换版本后请重跑本步。

### B. 小范围发现 + **免费 CPU LOO**

```bash
python -u scan.py --traits hostility --layers 13-14 \
  --max-probes 3 --top-k 64 --out out/local_v5 2>&1 | tee local.log

python stability.py --scan-dir out/local_v5 --trait hostility
```

`stability.py` 报 Top-8/64/256、out-of-fold 预测符号和 probe 支配度。LOO 可用但受仅 3 条发现探针的高方差影响。

### C. 排名与符号独立验证

```bash
python -u verify.py --scan-dir out/local_v5 --trait hostility \
  --top-k 8 --random-k 8 --dose 0.05 --generation-top 2 \
  --include-evil-pair --out out/verify_local_v5 2>&1 | tee verify.log

python report.py --verify out/verify_local_v5
```

主比较：Gradient Top-K vs **同层随机 + 自身梯度符号**。使用相同随机坐标和母候选符号的是符号诊断臂，**不是另一次独立随机抽样**。检验配对的差值 = (同层 Top 候选 heldout 效应) − (对应随机自身符号 heldout 效应)。

### D. **同输入**梯度/有限差分校准

```bash
python -u calibrate.py --scan-dir out/local_v5 --trait hostility \
  --split heldout --max-probes 1 --top-k 4 --dose 0.05 \
  --out out/calibrate_local_v5

cat out/calibrate_local_v5/summary.json
```

此处可分别检验局部 Taylor 导数 `(J(+d)-J(-d))/(2d)` 与当条 heldout prompt 的 `g`，避免将发现 prompt 到 heldout prompt 的变化混称有限差分误差。

### E. 根据资源晋级门槛决定是否扩大

```bash
python gate.py --scan-dir out/local_v5 --verify-dir out/verify_local_v5 \
  --trait hostility --out out/gate_local_v5.json
```

门槛通过后方可触发严格全层全特质扫描：

```bash
python -u scan.py --traits all --layers 0-31 --max-probes 3 --top-k 64 \
  --gate-scan-dir out/local_v5 --gate-verify-dir out/verify_local_v5 \
  --out out/global_v5 2>&1 | tee global.log
```

`gate.py` 失败退出码为 2。不建议覆盖门槛；若因方法论研究确需扩大，可以显式加 `--override-exploratory-gate`，该操作记入 scan manifest。

## 4. 额外实验

```bash
# 选择任务：先按 task 平均镜像次序；n=3/task-type 仅描述性分析
python -u choices.py --scan-dir out/global_v5 --top-k 2 --out out/choices_v5
cat out/choices_v5/summary.md

# 跨特质矩阵：其输入不同于 discovery，但其哈希已在 scan 时预注册
python -u specificity.py --scan-dir out/global_v5 --top-k 2 --out out/specificity_v5

# 组合非加性（仅假定目标函数下的局部交互，不等于人格协同）
python -u compose.py --scan-dir out/global_v5 --trait hostility \
  --top-k 4 --out out/compose_v5

# 不同嵌套上下文长度
python -u context_sweep.py --scan-dir out/global_v5 --trait hostility \
  --lengths 128,384,1024 --out out/context_v5
```

任何脚本读取 `probes/contrasts.jsonl` 时都会比较 SHA256。`choices.py` 与 `specificity.py` 分别也会检查自己的数据集是否与 **scan-time** 提前锁定的哈希一致。若发现后修改了任一数据，必须在**新输出目录重新发现/验证**；不能以改 `--out` 绕过。

如果使用自定义发现探针，`choices.py`、`specificity.py` 应追加 `--discovery-probes /path/to/your-original-contrasts.jsonl`；`scan.py` 在最初调用时同时传 `--specificity-probes ... --choices-tasks ...`，保证辅助验证数据也预先锁定。

## 5. 哈希性能控制

默认 `require_same_model` 每次重新计算 safetensors SHA256，是**最严格**、但消耗 I/O 的方案。若权重存储可靠、你只做探索，可以在 **verify/calibrate/choices/specificity/compose/context_sweep** 加 `--reuse-fingerprint`。首次扫描会保存可信的全量 SHA256 缓存；后续发现文件 metadata 不变则重用。更改 size/mtime/inode 会触发重新 hash；相同大小且伪造元数据的攻击不一定被发现。不要将该缓存宣称为抗篡改签名。

清单完整性：`sha256sum -c CHECKSUMS.sha256`；权重指纹位于每次扫描 `manifest.json`。模型 SHA 文件存在独立验证价值，但用户可编辑的目录和 manifest 不是不可篡改审计系统。

## 6. 当前最重要的科学边界

- 小样本估计很不稳定。配对均值应结合 SD/IQR、每对差值和准确的显著性检验；**符号检验不自动解决提示词和候选挑选偏差**。
- Local-gradient sensitivity 只能预测小剂量下的 teacher-forced likelihood 变化，不能直接说明存在真实、稳定的贪婪/胆小等人格。
- 真实选择与自由生成也只是标准化情境指标，不应替代跨任务能力保持率、人工盲评及长程行为评估。
- backward 已在本机实测通过（见 §3 A 的时延数字），但尚未覆盖 `generate`、batch>1、`--max-seq` > 384。换机或换版本后先重跑冒烟；未通过就保存 `smoke.log` 与 errors，不要进入 Stage E。

## 7. 统计单位、模式版本与资源晋级

**Choices**：同一任务的 A/B 位置互换是重复测量，不是独立样本。选择率与无效回答率先在每个 task 内平均两种顺序，然后计算跨 task 平均和 SD。当前每类只有 3 个独立任务，`summary.json` 中 `inferential_p_value=null`；原有 `n_total_paired=6` 仅是顺序化记录数，**不能作为统计 n**。不把五个 trait 当 15 个同质独立任务去进行总体假设检验。

**Gate**：7/8 + LOO Top-64 overlap ≥0.5 只是决定是否值得多租 GPU 的探索性启发式，不是对新场景的统计确认。八个神经元坐标共享 heldout prompts，不能假设独立性；Gate 的 p 值不作为通过条件。缺少 LOO 文件时会给出生成命令。

**Schema**：`gradient-scan-v2`、`verification-v2` 等名称代表稳定的**磁盘格式**，不代表软件发布版本；`package_release` 自 v4 起写入 manifest（当前 `v5.1`）。源码指纹用于隔离不同实现，不能用旧版本的扫描结果冒充新版本的发现结果。仅在格式本身发生不兼容变化时才升级 schema。

## 8. 可选的 Git 仓库迁移（不会自动修改你的 repo）

如果你的仓库含有 `01_packages/*.zip`、`00_README.md` 与 `.gitignore`：

```bash
# 先查看拟执行的操作；默认绝不修改仓库
python tools/migrate_repo_packages.py --repo /path/to/your/research-repo/qwentest

# 仔细核对输出后才允许落地
python tools/migrate_repo_packages.py --repo /path/to/your/research-repo/qwentest --apply

git -C /path/to/your/research-repo status --short

# 检查重复：同一版本族的 v1/v2/v3/v4 可保留完整文本；跨版本族完全相同的文件报告为冲突
python tools/dedup_check.py --workspace-root /path/to/your/research-repo/qwentest
```

脚本只迁移 ZIP 中合法 UTF-8 文本，拒绝路径穿越/ZIP symlink，跳过显见二进制；目标文件内容冲突直接报错，不覆盖。原 ZIP 保留在硬盘但在 `.gitignore` 中忽略，自动清理 `!01_packages/qwen-persona-matrix.zip` 例外；更新一个独立受管理的 README 校验表，不会擅自重排你原有的其他 MD5 表。只通过 `--apply` 写入，也不执行 Git add/commit。


### 迁移器的安全语义

- `--repo` 指 **直接包含 `01_packages/` 的工作区根目录**（如 `<repo>/qwentest`），不是包含 `qwentest` 的 Git 根目录；默认仅 dry-run。
- 工件完整性优先：v1/v2/v3/v4 各自解包，不删除跨版本的相同文本。`tools/dedup_check.py` 按 ZIP 名的版本后缀识别家族，同族重复允许，跨族重复默认返回 2（仅报告，不删文件）。
- 旧 ZIP 专用 MD5 表在可明确识别时替换为受管理的哈希表链接；混有非 ZIP 文件的表拒绝自动改写，避免误删既有审计资料。
- `.gitignore` 只清理可识别的旧 ZIP 例外及其专属注释；不相关注释保留。连续两次 `--apply` 应得到完全相同的文件内容。迁移只在上传仓库的**副本测试通过后**于原仓库运行；本包没有直接改动你的 Git 仓库。
- ZIP 二进制不入 Git 不等于原始 ZIP 可逐字节复原；请另行保存归档本体和 SHA256。

## 9. 本机首次实测记录（2026-10-27，hostility，13–14 层）

只用于说明工具的可用性与量级，**不构成结论**；同一批数据已由 `gate.py` 判为不通过。

| 阶段 | 结果 |
|---|---|
| A backward | 跑通；首条 66.0 s（首次编译），缓存后 0.30–0.92 s/条；无 traceback |
| B 发现（3 探针）+ LOO | top-64 重合 **0.417**、top-8 **0.292**；折外符号命中 0.375–0.75（≈ 机会水平） |
| C 验证（8 top + 8 随机 × 6 heldout） | Top-K **+0.0813** vs 随机自身符号 **+0.0298**；配对 **7:1:0**，精确符号 p=0.0703、符号翻转 p=0.0156 |
| D 同输入 Taylor（4 候选，dose 0.05） | 斜率 **1.043**、Spearman **1.0**、MAE 0.183（最大候选 g 4.13 vs FD 4.26） |
| 旧 Evil Pair | `L14N8949+` 绝对排名 7314/18432、`L13N7956−` 17967/18432，符号均不符 |
| `gate.py` | **pass=false**，唯一原因：LOO top-64 0.417 < 0.5 |

## 10. R2：语言质量读数 J2（v6 新增）

### 10.1 目标与梯度

$$J_2(x)=\log\Big(\tfrac1{|\mathcal L^+|}\sum_{y\in\mathcal L^+}e^{S(y)}\Big)-\log\Big(\tfrac1{|\mathcal L^-|}\sum_{y\in\mathcal L^-}e^{S(y)}\Big),\qquad S(y)=\tfrac1T\sum_t\log P(y_t\mid x,y_{<t})$$

$$\frac{\partial J_2}{\partial d}=\sum_{y}\omega(y)\,\frac{\partial S(y)}{\partial d},\qquad \omega=\mathrm{softmax}_{\mathcal L}(S)$$

因为 `∂/∂d log Σ e^S = Σ ω ∂S`，权重 ω 在求导点是**常数**，所以**一次** grad-enabled 批量前向就给出精确一阶导（无采样、无近似）。槽位词必须单 token、两侧等长，两者都在运行期断言。

### 10.2 词表：从人物指向的公开语料挖，不手写

- 语料 **Jibes & Delights**（Sodhi/Pant/Mamidi, WOAH 2021, `aclanthology.org/2021.woah-1.14`；仓库 `ravsodhi/jibes-and-delights`，**Apache-2.0**）：RoastMe 121,035 条人物指向嘲讽 / FreeCompliments+ToastMe 91,149 条人物指向赞美。
- `mine_slots.py`：帧匹配 → WordNet 词性 → 前导空格单 token → 显式排除脏话/代辱语 → `h≥5 且 h>1.5n`（对照侧对称）。**帧合法性由「该词确实出现在这个帧里」保证**。
- 随包提供 `lexicons/jibes_adj_verb_pools.json`（**只含派生词表，不含语料原文**，附 attribution）：ADJ `You are so {w}.` **19/19**、VERB `You can't even {w}.` **14/14**，各含互斥 A/B 两半。
- **NOUN 帧被否**：同一帧在 RoastMe 里挖出的是身份词（boy/dude/lesbian/gay/white），不是侮辱 —— 已从池中剔除并记录在 JSON 里。
- HurtLex/NRC 路线（CC BY-NC-SA / 非商用）保留在 `build_lexicon.py`，**外部文件、不入仓**。

### 10.3 用法

```bash
# 用随包词表扫描（9 条 discovery，13-14 层）
python -u scan_sem.py --probes probes/contrasts_v2.jsonl --max-probes 9 --layers 13-14 \
  --top-k 256 --lexicon lexicons/jibes_adj_verb_pools.json --out out/j2_v6

# 读数自身的 held-out 效应（J2 度量，top-K vs 同层随机自身符号）
python -u verify_sem.py --scan-dir out/j2_v6 --probes probes/contrasts_v2.jsonl \
  --top-k 8 --random-k 8 --dose 0.02 --out out/verify_j2_v6

# 读数稳定性（词表互斥对半 A/B；prompt 互斥对半）
#   两次 scan_sem 用 --pool-variant A / B，再比较 *_all_gradients.pt
```

`scan_sem.py` 产出的产物模式与 `scan.py` **完全相同**（`manifest.json` schema `gradient-scan-v2`、`calibration.json`、`*_top.json`、`*_all_gradients.pt`），所以 `stability.py` / `report.py` 可直接读；`verify_sem.py` 用 J2 自身度量，`verify.py` 仍用字符串对比（J1），两者不可混用。

### 10.4 本机实测（2026-10-27，RTX 4090，13–14 层）

| 量 | 值 |
|---|---|
| 探针互斥拆分（9 条 discovery） | 前 4 vs 后 5：r=0.487、J64=0.347；偶/奇：r=0.633、J64=0.488 |
| 对照：J2 只用 6 条（3v3） | r=0.323、J64=0.143 |
| 对照：J1（3v3） | r=0.03、J64=0.03 |
| 词表互斥对半（6 条 prompt） | r=0.583、J64=0.267（HurtLex 池 0.753/0.320 —— 池子选择是二阶的） |
| held-out 效应（`verify_sem.py`，dose 0.02，6 条未参与筛选） | top **+0.01483** vs 随机 **+0.00477**，配对 **8 胜 0 负**，精确符号 p=**0.0078** |
| 同输入中央差分（4 候选） | 符号 4/4 一致；幅度比 0.66–1.77（J1 是 3–4%，所以 J2 只用符号与排序，不用幅度预测） |

### 10.5 已知限制（不要越过这些说话）

- **分族不对称**：效应几乎全在 ADJ（+0.0265），VERB 只有 +0.0032；`scan_sem` 的族间等权平均把这一点抹平了。按族分别排名或加权属于下一版。
- **9 条 prompt 是 post-hoc**：它们是看到 6 条结果之后写的，所以 9 条上的复现率不是预注册结论。确认需要另一批独立撰写的 prompt。
- **稠密度不变**：top-64 仍只占 `sum|g|` 的 1.7%，所以 top-K 是**筛选器**，不是「找到了那几个神经元」。
- `semantic.py` 的 FD 检查是**同一 prompt、同一语言**的一阶一致性，不是行为效度。
- 诊断用的 `_diag6_hostility.jsonl`（把 heldout 临时重标为 discovery）**不随包**；如需复现：复制 `probes/contrasts.jsonl`，把 6 条 hostility heldout 的 `split` 改成 `discovery` 即可。

### 10.6 v6 新增文件

| 文件 | 作用 |
|---|---|
| `semantic.py` | J2 值与精确加权 VJP（批量、一次前向）；等长/单 token 断言；`fd_check` |
| `scan_sem.py` | J2 扫描；产物模式与 `scan.py` 相同，可被 `stability.py`/`report.py` 直接读 |
| `verify_sem.py` | 用 **J2 自身度量**做 held-out 效应检验（top-K vs 同层随机自身符号） |
| `mine_slots.py` | 从人物指向的公开语料挖槽位（帧匹配 + WordNet 词性 + 单 token + 排除清单） |
| `build_lexicon.py` | 备用路线：HurtLex/VADER/WordNet 组池（外部文件，不入仓） |
| `lexicons/jibes_adj_verb_pools.json` | 派生词表（ADJ 19/19、VERB 14/14，含 A/B 两半）+ 来源与许可声明 |
| `probes/contrasts_v2.jsonl` | 原 45 行 + 6 条新 discovery（hostility 现为 9 discovery + 6 heldout） |
| `tests/test_semantic.py` | 5 项 CPU 测试：批量与逐条一致、等长拒绝、语言质量恒等式、梯度对中央差分、J2 有限性 |
