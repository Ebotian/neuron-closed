# AGENTS.md — 维护规范（本仓）

本仓是**方法/仪器仓**：它回答"在哪几个 (层, 神经元) 上注入、注入多少"，并把每个数字的证据链留在仓内。
它不是"开源世界获取能否改进搜索"那个主题仓（那在 `rsi-dev`）。

## MUST（不可协商）

1. **判据先冻结**。任何实验在跑之前，把判据写进脚本 docstring 或 `doc/plan-*.md`；跑完不改判据，除非另起一节写明"为什么改、改前是什么"。
2. **数字先进 `doc/metadata.json`，再 `make meta && make check`，然后提交**。文档里出现的数字必须能在 `results/` 的 JSON 里找到。
3. **三种来源分开写**：① 论文/仓库的原话（逐字引，带版本）；② 我们的转述或推导；③ 我们的判断。三者不得混在一句话里。
4. **钉版本**：论文用 `check_citations.py`（arXiv 编号 + 版本），代码仓用 commit SHA，数据快照用 sha256。未钉版本的引用不得进结论。
5. **声明核查范围**：每个结论段末尾写清"核了哪些 ID/版本/文件，没核什么（例如未重跑 GPU）"。
6. **不可比就不比**：跨量化的 p50、跨约定的剂量、跨判据的绝对值，一律不得并列。
7. **零 GPU 也能验**：`make check` 必须 stdlib-only（无网络、无 GPU、无 torch）。

## 结构

```
doc/            metadata.json（单一数字来源）、plan-*.md（科学记录）、references/（外部证据）、NEXT_SESSION.md
results/        GPU/远端运行的原始产物（JSON 为主，MD 为渲染）
scripts/        薄壳实验脚本（≤120 行）+ core/ estimators/ analysis/ 三层模块（见 doc/ARCHITECTURE.md）
tools/          check.py（唯一验证入口）、evidence_ledger.py、dedup_check.py
vendor/         冻结的测量包（带 CHECKSUMS.sha256，不修改）
lexicons/       词表（只存可再分发的；第三方来源见 NOTICE）
probes/         上下文/对比集
evidence/       外部证据总账 ledger.json
lean/           Lean 4 形式化（计划 2.44 的代数；用本机公共 mathlib 环境，见 NEXT_SESSION）
```

## 完成定义（Definition of Done）

一个变更算完成，当且仅当：

- `make check` 全绿；
- 新数字在 `doc/metadata.json` 里有对应条目，且 `results/` 里有产物；
- 结论段写清判据、样本量、误差尺度、边界；
- 若推翻了本仓旧结论，**保留旧结论并标注"已被 X 取代"**，不删。

## 写作约定

- 科学记录用中文，代码与提交信息中的技术术语保留英文原文（如 `u = ŴB⁻¹d*`、`p50`）。
- 每条结论尽量带三个量：**效应**、**样本单位**、**误差尺度**。
- 表格优先；同一条结论只在一处给数，别处引用文件名。

## 不做

- 不提交模型权重、不提交第三方原始数据快照（只提交统计量与 sha256）。
- 不在 `vendor/` 上改代码（校验和会失效）；需要改动就新开版本目录。
- 不为"跑通"调阈值；阈值改动必须单独一节记录。
