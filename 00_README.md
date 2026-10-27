# qwentest — Qwen3.5-4B 神经元 steering / 人格矩阵实验 归档

整理时间：2026-10-27（会话结束、实例关机前）
整理范围：本次任务的全部包、结果、压缩包与工具。原始文件已从 `~/Downloads/` 移入本目录，未保留副本。

远端实例（2026-10-27 更换）：`root@connect.westd.seetacloud.com -p 11168`，AutoDL **RTX 4090（48 GB）**，
数据盘 `/root/autodl-tmp`。前一台为 `connect.westc.seetacloud.com:25785` / RTX 4080 SUPER（32 GB）；换机后数据盘与 `~/.triton` 缓存都保留（同 sm_89，所以首个 backward 不再重编 66 s），
但**数值不再逐位相同**：`p50` L13 9.737809181213379 / L14 10.338690757751465（4080 上为 9.752792358398438 / 10.32901382446289，差 ~0.15%），
`hostility_01` margin −0.8584（4080 −0.8630），top-12 排序不变、`|g|` 差 <1%。因此：
**同一台机同一次栈可复现；跳机只保证排名与量级，不保证末位数字。**
旧 `out/` 与 `manifest.json` 里的 `runtime.gpu_name` 为 4080，`verify.py` 的 `require_same_runtime` 会拒接一—这是设计行为，新实验请新开 `--out`。

---

<!-- auto:workstream-status:start -->
| 项 | 值 |
|---|---|
| 工作流 | `qwentest` |
| 版本 / 状态 | `0.4.0` / `draft`（2026-10-27） |
| 现役 | R2 — J2 readout; G2 passes at 19 discovery prompts (G1 0.524 / G2 0.471–0.662) |
| 路线 | `R0` done · `R2a` done · `R2b` done, minimal · `R2` active · `R1` todo · `R3` todo · `R4` todo · `R5` todo |
| 判据 | `G1` · `G2` · `G3` · `G4` · `G5`（定义见 `metadata.json`，说明见 `05_plan-gradient-discovery.md` §1） |
| 实测 | G1 r=0.693/J64=0.524（19 条 prompt） · G2 r=0.668–0.781 / J64=0.471–0.662（9v10 / 10v9） · verify 8/8 p=0.0078（6 reused probes -> replication, not confirmation） |
<!-- auto:workstream-status:end -->

## 工程约束（一条命令：`make -C qwentest check`）

本目录的**每一条对外声明都必须能被重新推导**，而不是靠记忆。数字与状态只有一个来源，其余都是渲染出来的。

| 规则 | 机制 |
|---|---|
| 版本 / 状态 / 判据 / 路线 / 实测数字 | **唯一来源 `metadata.json`**；`00_README.md` 里的状态块是渲染结果（`make meta` 重渲染，`make check` 校验） |
| 包清单（md5 / sha256 / 文本文件数） | 由 `01_packages/*.zip` 现算后渲染进受管段；`make check` 校验区块与磁盘一致 |
| zip 与解包树 | 逐成员比对（`ziptrees`）；不等即失败 |
| 包内自校验 | 每个 `03_unpacked/*/<root>/CHECKSUMS.sha256` 必须验过 |
| 代码 | 全部 `.py` 可编译（`compile`） |
| 测试 | 免 torch 的单测必须通过；其余标为 skipped 并提示到 GPU 机跑 |
| 去重 | 版本族内允许重复，**跨族重复即失败**（`dedup`，只读不删） |
| Git 边界 | 任何 `*.zip / *.tar.gz / *.pyc / *.pt` 被跟踪即失败（`git`） |

- 入口：`make -C qwentest check`（stdlib-only，无网络、无 GPU、无 torch）；分项可用 `make -C qwentest blocks|ziptrees|checksums|compile|tests|dedup|git`。
- 钩子（opt-in）：`make -C qwentest hooks` → `core.hooksPath=.githooks`，`pre-commit` 只在 `qwentest/` 被暂存时跑 `make check`（约 2 s），非 qwentest 提交不受影响。
- 工具：`tools/check.py`（本文件的校验器）、`tools/migrate_repo_packages.py`（解包 + 受管段，仓库级**规范副本**）、`tools/dedup_check.py`（只报告，不删）。包内同名工具是随包冻结的副本，改动以 `tools/` 为准。
- 输入放在**冻结包之外**：新的探针/词表写到 `qwentest/probes/`、`qwentest/lexicons/`（本次新增 `probes/contrasts_v3.jsonl`），不要往 `03_unpacked/` 里加文件——那里由 zip 决定，加了就会让 `ziptrees` 与 `check-remote` 的树哈希失配（本日实测：误放一份探针到远端包目录，哈希从 `b23ee718…` 漂到 `e00c4de1…`）。
- 改数字的顺序：**先改 `metadata.json`（或重跑实验）→ `make -C qwentest meta` → `make -C qwentest check` → 提交**。手改状态块会被 `check` 抓住（已实测：改一个数字即 FAIL）。

## 目录结构

```
qwentest/
├── 00_README.md            ← 本文件
├── 01_packages/            实验包（zip，可上传运行）
├── 02_archives/            结果与重建归档（tar.gz）
├── 03_unpacked/            上述包/归档的解包内容（便于直接查看）
├── 04_tools/               与远端交互的辅助脚本
├── 05_plan-gradient-discovery.md  梯度发现的读数路线（R2 现役，R0/R1/R3/R4/R5 为 TODO）
└── scripts/                本地统计脚本（stats_single.py、stats_timing.py、dedup_check.py）
```

后续路线与预注册判据（G1–G5）见 **`05_plan-gradient-discovery.md`**：R2（token 上的正规语言精确边缘化）为现役工作，其余登记为 TODO；该文件先冻结判据、再跑实验。

---

## 01_packages — 实验包

Package checksums are maintained in the [Unpacked package provenance](#unpacked-package-provenance-automatically-maintained) table below.

### persona-matrix v2 相对 v1 的关键改动（都已落在这份包里）

- `--dose` 默认 **1.0**（与 v3 记录同一剂量标尺；`package_version: v2-p2`）
- `refusal` 强度按原作者注记改为 **+1.5**（原先 1.0，跨 node 剂量不可比）
- `zero_hook` 覆盖本次运行涉及的全部 (层, 神经元)，不再写死 L13/L14
- 新增 **norm-matched 随机方向对照** `randdir:L<层>:s<种子>`（残差空间随机单位向量，范数 = dose×p50）
- `analyze.py` 导出**真盲评表** `blind_*.jsonl` + 密钥 `blindkey_*.jsonl`（全 run 洗牌、去掉条件字段）
- `geometry()` 标注 `kind: identity` —— 同一神经元两端的 `-1` 是恒等式，不是发现
- 新增 `gate_check.py`（Gate 0/2/2b）与 `desire_interactive.py`（流式、Ctrl-C 打断）

---

## 02_archives — 结果与重建归档

| 文件 | md5 | 内容 |
|---|---|---|
| `qwen-results.tar.gz` | `2a96a1077dc691b7b4f77eb278d80fa3` | 早期两批 dose 1.0 结果（6 + 80 条） |
| `persona-single-d0.5.tar.gz` | `74175ef744a0c000d1f13b37be0c9f51` | persona-matrix dose 0.5 / phase single 全量：1080 条 |
| `rebuild-kit.tar.gz` | `8c52887f3e38b240a8e75c1e52a6bc32` | 重建包：`REBUILD.md` + `env.txt` + 代码 + 两个 run 的 out/ + gate 输出 + 各 zip |

---

## 03_unpacked — 解包内容

| 目录 | 说明 |
|---|---|
| `qwen-results/` | `out/results_3dca6cebd49e.jsonl`(6) + `out/results_5ddba4527220.jsonl`(80) + **`NOTE_random_pair_03.md`** |
| `persona-single-d0.5/out/` | `results_626721938e67f4.jsonl`(1080) + manifest + geometry |
| `qwen-evil-repro-v2/` `qwen-evil-repro-v3/` | 两个版本的包目录（仓库式结构可直接跑） |
| `qwen-persona-matrix-v2/` | v2-p2 包目录 |
| `rebuild-kit/` | 重建包解包（**已去重**，见下；只剩 `REBUILD.md`、`env.txt` 等 5 个别处没有的文件） |

`rebuild-kit/` 解包目录做过一次去重：只留下别处没有的 5 个文件——`REBUILD.md`、`env.txt`、
`qwen-persona-matrix/README.md`（这一版 README 与 v1、v2 都不同）、
`qwen-persona-matrix/gate_out/gate_72a7e1cb.json` 和 `gate_72a7e1cb.md`。删掉的 20 个文件
（17 个文本 + `qwen-evil-repro-v2.zip`、`qwen-persona-matrix-v2.zip`、`qwen-results.tar.gz`）
都与下列位置的文件逐字节相同，动删之前对过 md5；看内容请直接去这些位置：
`qwen-evil-repro-v2/`、`qwen-persona-matrix-v2/`、`qwen-results/out/`、`persona-single-d0.5/out/`、
`01_packages/`、`02_archives/`。`rebuild-kit.tar.gz` 没有改动，仍是完整的一套；要看整包结构就
`tar xzf 02_archives/rebuild-kit.tar.gz -C /tmp/kit`，不要解回本目录（否则副本又回来了）。
`scripts/dedup_check.py` 可以复核这次删除没有丢内容。

`NOTE_random_pair_03.md` 是本次唯一未结线索的登记：`random_pair_03`（L14 N2990 / L13 N4787）
在 dose 1.0 的 8 个 prompt 中有 4 次落入安全/越界语域（p04 作弊、p06 卷入犯罪、p07 人际遮掩、
p08 椅子那一侧），且跨任务复用同一短语 "face the problem/challenge"。若成立，它作为"随机对照"
会低估 target_pair 的特异性——需要一批正式随机对照才能定论。

---

## scripts — 本地统计（纯本地，不需要 GPU）

| 文件 | 说明 |
|---|---|
| `stats_single.py` | 单个 run 的描述统计：变化率、与 baseline 的相似度、退化率、按类型/上下文长度的计数 |
| `stats_timing.py` | 同一 run 的耗时分解：prefill vs decode 最小二乘、按上下文长度与条件分组 |
| `dedup_check.py` | 复核 `03_unpacked/rebuild-kit/` 里没有与别处逐字节相同的文件（默认只看不改，`--apply` 才删） |

两个 stats 脚本的默认输入与输出都按脚本自身位置解析（输入 `../03_unpacked/persona-single-d0.5/out/results_626721938e67f4.jsonl`，输出同目录 `out/`），在任意工作目录下都能直接跑；`out/` 是派生结果，不入 git。

---

## 04_tools

| 文件 | 说明 |
|---|---|
| `sshpass_pty.py` | pty 驱动的 ssh/scp 密码应答（本机无 `sshpass`）。用法：`python3 sshpass_pty.py '<密码>' ssh -p 25785 ...`。密码不写入文件，只作参数传入。 |

---

## 实验状态一览

| 项 | 状态 |
|---|---|
| `qwen-evil-repro` dose 1.0 | 2 批（6 + 80 条），**已逐条审阅翻译** |
| `persona-matrix` dose 0.5 phase single | **1080/1080 完成，尚未分析** |
| `gate_check.py` | 三项全 PASS：p50 差 0、`evil_A+evil_B@1.0` 对 v3 `target_pair` **8/8 逐字相同**、randdir hook 正常触发 |
| 结论 | 两个包共享同一剂量标尺，`evil_A/evil_B` 与 v3 的 `target_pair` 是同一条件 |
| 未做 | dose 1.0 的 single/pair 批量、正式随机对照、三轴盲评打分 |

### 关键锚点（重建时用来验证等价）

```
config.json sha256   ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670
p50 L11 9.133934020996094    L12 9.303200721740723
    L13 9.752792358398438    L14 10.32901382446289
    L18 13.89737319946289    L19 16.282630920410156
环境 python 3.12.3 / torch 2.8.0+cu128 / transformers 5.20.2 / tokenizers 0.23.3
     fla-core 0.5.2 / causal-conv1d 1.7.0 / accelerate 1.15.0 / numpy 2.3.2
```

### 下次开机的三条命令

```bash
cd /root/autodl-tmp/qwen-persona-matrix
/root/miniconda3/bin/python -u gate_check.py \
  --expected /root/autodl-tmp/qwen-evil-repro/out/results_5ddba4527220.jsonl   # ~4 分钟
/root/miniconda3/bin/python -u desire_interactive.py                            # 流式，Ctrl-C 打断
/root/miniconda3/bin/python analyze.py out                                      # 0.5 那批的表 + 盲评表
```

注意：conda 不在非交互 ssh 的 `PATH` 里，必须用 `/root/miniconda3/bin/python` 绝对路径。

<!-- auto:unpacked-packages:start -->
## Unpacked package provenance (automatically maintained)

Binary ZIP files remain ignored; text versions are under `03_unpacked/`.

| Archive | MD5 | SHA256 | Unpacked text files |
|---|---|---|---:|
| `qwen-evil-repro-v2.zip` | `d59e0f07c0c60f47c3f0db20827e9c6d` | `c605d542a558faa65ccc5724f42a60634126119a78a8eabe452ca2e2457f4489` | 3 |
| `qwen-evil-repro-v3.zip` | `5933e17a7f9681de937632b55d71192e` | `241d2b74f6379a08b42411e32bb3f2e7f39b5937c49644eecdb526a05e60ead8` | 3 |
| `qwen-gradient-discovery-v2.zip` | `dc69d58dde3d6b68d954d6b9c1628799` | `f5a607189eeed778459f79a743aae3d4cdf3a415d413112455a9e38a55c3ce32` | 20 |
| `qwen-gradient-discovery-v3.zip` | `621845f982a007ac2aa517168670e0e1` | `e80c314a1b3a1b1459be94d27848c333bc77b5f8e0dba2573a4d0a948195d5b2` | 23 |
| `qwen-gradient-discovery-v4.zip` | `527775f4226848a1644ce73fd6f128d5` | `3a48a562cb9993a000b5da84c34e5f5047b96b01eb169d8abc6c3d65dfad9e6c` | 25 |
| `qwen-gradient-discovery-v5.1.zip` | `63c754d21f55a7c8c6c96fa08290387f` | `be1b21f3e697c077d8de5a98e7b5b5b1c52bb83846fcfceb843a4b0570c3cca6` | 27 |
| `qwen-gradient-discovery-v5.zip` | `0d38ab8b7ac3a5b7de445662ad2d3f04` | `cda8ee267e58dcfdd5a03d1780c912e27472ebedbfcbd084cbf55df7b4086606` | 27 |
| `qwen-gradient-discovery-v6.zip` | `49cfc2d6917b4e0215e15f6ef3838ba8` | `90e584e0e1a8c51e50b62935b6abf291a2f5481f8ea4e7f6844537dbf35169a5` | 35 |
| `qwen-gradient-discovery.zip` | `802746e8ea64ea2de880802d967246c7` | `85a3e56b5953911d15c64a29e0d9a6cb1e7933d216deca5fc4f02f77781669dd` | 9 |
| `qwen-persona-matrix-v2.zip` | `96e71ce2063c7f785230f1a09d9d090c` | `5561d542ba706a24553563443461f074066cb02194d19b0c5072842b25c15341` | 8 |
| `qwen-persona-matrix.zip` | `b9d57f06d88695417c92810b6c0d6630` | `9576b594b8d9d4da710586b742e334aad5b9b4a170c2449f0639db23e92b3793` | 5 |
<!-- auto:unpacked-packages:end -->
