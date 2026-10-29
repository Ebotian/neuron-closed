# Qwen3.5-4B White-box Gradient Discovery — v3 (审查修订)

本包是针对冻结 Qwen3.5-4B 的可微激活归因、候选筛选与因果检验工具。**不训练、不更新模型权重**。搜索目标是少量人写的续写对比 `J=mean logP(y+|x) − mean logP(y-|x)`，故证据首先支持的是 *language proxy*，不是人格本体。

**版本边界：** 源码已在 CPU 环境通过 24 项数值、统计与完整性测试；未在 RTX 4080 SUPER 上验证本版的 Qwen3.5 **backward**。不能把已通过的 forward/FLA import 当作 backward 已验证。v2 的扫描 manifest 不兼容 v3 源码指纹；首次 GPU 运行必须新开 `--out`。

## 1. 本轮修复与剩余限制

| 审查问题 | v3 处置 |
|---|---|
| `check_scan_file` 死代码 | `verify/calibrate/compose/context_sweep` 真实调用；`specificity/choices` 也校验**原始发现文件**，同时独立校验在扫描开始时预登记的辅助测试集 SHA256。缺少发现哈希直接拒绝。修改 heldout/试题后需重新扫描，不能换 `--out` 混用。 |
| 小样本只报告点估计 | `report.py` 增加配对差 SD/IQR、中位数、95% bootstrap 描述区间、双侧精确符号检验，以及 n≤18 的双侧精确符号翻转检验。独立统计单元为唯一匹配的物理神经元坐标。 |
| Choices 未报告配对效应和失效率 | 按相同 task × A/B order 配对 baseline/arm，主报总分母 `ΔP(选中目标项)` 及 `Δinvalid rate`；另报双方有效的 complete-case 结果、双方无效/仅一方无效的频数，明确选择偏差风险。缺行/重复行拒绝总结。 |
| 把迁移误差当同输入校准 | 验证报告主表只列观察效应、SD/IQR； discovery→heldout MAE/slope 被单独标作**迁移诊断**，见 JSON；同输入有限差分只由 `calibrate.py` 单独报告。 |
| 没有噪声底 | 同层随机自身符号分布 SD/IQR/经验 Q95、Top 超出随机经验 Q95 的比例。8 个随机坐标的 Q95 极不精确，只是描述性参考。 |
| CHECKSUMS 不完整 | ZIP 中的所有正规文件（包括 README、TROUBLESHOOTING、tests、tools）均纳入 `CHECKSUMS.sha256`，**仅校验清单自身例外**（否则存在自引用哈希问题）。 |
| 每个程序重读 8.8GB | 默认依然逐字节 SHA256；加 `--reuse-fingerprint` 可选择缓存模式：首次正常 SHA256，后续仅对比路径、文件集合、size、mtime_ns、inode/dev 后重用已存哈希。**这是性能与防篡改保证的显式权衡**，无法抵抗蓄意恢复全部文件元数据的修改。manifest 标记是否启用缓存。 |
| 无预注册晋级判据 | 新增 `gate.py`：**≥8 个非零的独立配对、正差比例≥7/8、3 条以上 discovery probes 且 mean LOO Top-64 overlap≥0.5**。完整的 `--traits all --layers 0-31` 默认受此门槛约束；若显式 `--override-exploratory-gate` 可绕过，但会写入 manifest。 |
| Git ZIP 忽略/源码不可追踪 | `tools/migrate_repo_packages.py`（dry-run 默认）可将 `01_packages/*.zip` 中安全的文本解到 `03_unpacked/`、统一 `.gitignore`、在 `00_README.md` 创建受工具管理的 MD5/SHA256 表。**没有修改用户仓库；只有用户指定 `--repo ... --apply` 才执行。** |

**数理提醒：** 8 对里 7:1 的双侧精确符号检验 `p=0.0703125`，8:0 才是 `p=0.0078125`。**7/8 是资源控制的探索性晋级条件，不是 0.05 统计显著性条件**。有多个 trait、多种实验和多次选择，想要确认性结论还须独立预注册数据与多重比较校正。即使 8:0 通过也不能证明稳定人格。

## 2. 模型和环境

你当前 AutoDL 环境：Python 3.12.3、torch 2.8.0+cu128、Transformers 5.20.2、Triton 3.4.0、FLA 可导入、causal-conv1d 1.7.0、RTX 4080 SUPER (租赁实例报告约 32GiB 显存)。请沿用环境，不升级 CUDA/Torch。默认模型路径 `/root/autodl-tmp/Qwen3.5-4B`。

```bash
cd /root/autodl-tmp
unzip -o qwen-gradient-discovery-v3.zip
cd qwen-gradient-discovery-v3
sha256sum -c CHECKSUMS.sha256
python -m unittest discover -s tests -v
```

## 3. 推荐的最小实验流程（A → B → C → D）

### A. 检查 backward kernel 可用性

```bash
python -u scan.py --traits hostility --layers 13-14 \
  --max-probes 1 --top-k 32 --out out/smoke_v3 2>&1 | tee smoke.log
```

这一步仍是重要未知数。报错保留在 `out/smoke_v3/errors/*.traceback.txt`；切勿仅因报错盲目升级 Torch/Triton。

### B. 小范围发现 + **免费 CPU LOO**

```bash
python -u scan.py --traits hostility --layers 13-14 \
  --max-probes 3 --top-k 64 --out out/local_v3 2>&1 | tee local.log

python stability.py --scan-dir out/local_v3 --trait hostility
```

`stability.py` 报 Top-8/64/256、out-of-fold 预测符号和 probe 支配度。LOO 可用但受仅 3 条发现探针的高方差影响。

### C. 排名与符号独立验证

```bash
python -u verify.py --scan-dir out/local_v3 --trait hostility \
  --top-k 8 --random-k 8 --dose 0.05 --generation-top 2 \
  --include-evil-pair --out out/verify_local_v3 2>&1 | tee verify.log

python report.py --verify out/verify_local_v3
```

主比较：Gradient Top-K vs **同层随机 + 自身梯度符号**。使用相同随机坐标和母候选符号的是符号诊断臂，**不是另一次独立随机抽样**。检验配对的差值 = (同层 Top 候选 heldout 效应) − (对应随机自身符号 heldout 效应)。

### D. **同输入**梯度/有限差分校准

```bash
python -u calibrate.py --scan-dir out/local_v3 --trait hostility \
  --split heldout --max-probes 1 --top-k 4 --dose 0.05 \
  --out out/calibrate_local_v3

cat out/calibrate_local_v3/summary.json
```

此处可分别检验局部 Taylor 导数 `(J(+d)-J(-d))/(2d)` 与当条 heldout prompt 的 `g`，避免将发现 prompt 到 heldout prompt 的变化混称有限差分误差。

### E. 根据资源晋级门槛决定是否扩大

```bash
python gate.py --scan-dir out/local_v3 --verify-dir out/verify_local_v3 \
  --trait hostility --out out/gate_local_v3.json
```

门槛通过后方可触发严格全层全特质扫描：

```bash
python -u scan.py --traits all --layers 0-31 --max-probes 3 --top-k 64 \
  --gate-scan-dir out/local_v3 --gate-verify-dir out/verify_local_v3 \
  --out out/global_v3 2>&1 | tee global.log
```

`gate.py` 失败退出码为 2。不建议覆盖门槛；若因方法论研究确需扩大，可以显式加 `--override-exploratory-gate`，该操作记入 scan manifest。

## 4. 额外实验

```bash
# 选择任务：带顺序镜像及“无效回答”成对混杂审计
python -u choices.py --scan-dir out/global_v3 --top-k 2 --out out/choices_v3
cat out/choices_v3/summary.md

# 跨特质矩阵：其输入不同于 discovery，但其哈希已在 scan 时预注册
python -u specificity.py --scan-dir out/global_v3 --top-k 2 --out out/specificity_v3

# 组合非加性（仅假定目标函数下的局部交互，不等于人格协同）
python -u compose.py --scan-dir out/global_v3 --trait hostility \
  --top-k 4 --out out/compose_v3

# 不同嵌套上下文长度
python -u context_sweep.py --scan-dir out/global_v3 --trait hostility \
  --lengths 128,384,1024 --out out/context_v3
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
- 一些后向内核仅经 forward 验证；真实 Qwen3.5 GPU backward 尚未实测。未通过冒烟请保存 `smoke.log` 与 errors 文件，不要直接进入 Stage E。

## 7. 可选的 Git 仓库迁移（不会自动修改你的 repo）

如果你的仓库含有 `01_packages/*.zip`、`00_README.md` 与 `.gitignore`：

```bash
# 先查看拟执行的操作；默认绝不修改仓库
python tools/migrate_repo_packages.py --repo /path/to/your/research-repo

# 仔细核对输出后才允许落地
python tools/migrate_repo_packages.py --repo /path/to/your/research-repo --apply

git -C /path/to/your/research-repo status --short
```

脚本只迁移 ZIP 中合法 UTF-8 文本，拒绝路径穿越/ZIP symlink，跳过显见二进制；目标文件内容冲突直接报错，不覆盖。原 ZIP 保留在硬盘但在 `.gitignore` 中忽略，自动清理 `!01_packages/qwen-persona-matrix.zip` 例外；更新一个独立受管理的 README 校验表，不会擅自重排你原有的其他 MD5 表。只通过 `--apply` 写入，也不执行 Git add/commit。
