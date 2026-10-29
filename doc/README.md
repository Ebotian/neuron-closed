# 文档索引与约定

本目录是**唯一的文字来源**。`results/` 放产物，`doc/` 放解释；两者由 `make check` 交叉校验。

## 约定

1. **数字只在 `metadata.json`**：正文里出现的数字若不是从 `results/` 的 JSON 生成，就必须在 `metadata.json` 里有对应条目。
2. **每一节都要有判据**：写在运行之前；若事后修改，另起小节写"改前/改后/理由"。
3. **三种来源分开**：① 论文/仓库原话（逐字，带版本）② 我们的转述或推导 ③ 我们的判断。
4. **被否的结论保留**：写"已被 X 取代"，不删。本仓最值钱的记录是那些**被否掉的预测**。

## 索引

| 文件 | 说明 |
|---|---|
| `plan-gradient-discovery.md` | 科学记录主文件（§1 判据与基线；§2 每一节实验；§3 TODO；§4 明确不做；§5 记录方式） |
| `plan-specificity-dose.md` | 特异性与剂量的早期计划（06） |
| `metadata.json` | 版本 / 状态 / 判据 / 路线 / 实测数字（渲染到 README） |
| `NEXT_SESSION.md` | 交接单：当前状态、下一步、远端注意事项 |
| `references/qwen3-8b-neuron-atlas.md` | 第三方 Qwen3-8B 单神经元图谱的重算与影响分类 |
| `references/evil-qwen-mlx-replication.md` | 社区 evil-qwen 的 MLX 复现记录 |
| `references/qwen35-community-atlas.md` | Qwen3.5-4B 社区图谱：配方库、测量协议、发布/未发布清单、三路比对 |
| `references/preprints.md` | 四篇预印本的钉版记录与它们改变了什么 |
| `references/CANS-2610.35530-v1.md` | CANS 正文精读（TeX 源）：公式对照、超参数、可逐字引用的边界 |
| `references/synthesis-external-evidence.md` | 8 来源 × 14 主张的机械校验总账（矩阵由 `evidence/ledger.json` 渲染） |
