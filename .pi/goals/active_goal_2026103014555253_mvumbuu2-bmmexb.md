{
  "version": 3,
  "id": "mvumbuu2-bmmexb",
  "objective": "把现役测量链（p_a2_pathj2 / p_a_reinforce / noise_scale / p_a4_closed_form / reliability / fd_gate / quotient_compare_v2）拆成 core/estimators/analysis 三层模块，每一步都用逐值回归证明行为不变，修掉重构过程中暴露的 bug，并把 8×32 基准压到 GPU 均值 ≥60% 且墙钟 ≤60 s。",
  "status": "active",
  "autoContinue": true,
  "usage": {
    "tokensUsed": 214110,
    "activeSeconds": 215
  },
  "sisyphus": false,
  "revision": 24,
  "createdAt": "2026-10-30T06:55:52.538Z",
  "updatedAt": "2026-10-30T06:59:38.358Z",
  "scheduler": {
    "version": 1,
    "owner": "01a186de-214e-758c-8987-7241e0406887",
    "generation": "0520d339-c2e6-4141-8eea-3056fba3b2c6",
    "used": 1,
    "phase": "running",
    "repairUsed": false,
    "decision": {
      "kind": "ready",
      "purpose": "kickoff"
    },
    "dispatch": {
      "id": "f572f2c0-92c0-48db-bc2b-8f6961c4b9df",
      "kind": "kickoff",
      "claimedAt": 1793343352551
    }
  },
  "taskList": {
    "tasks": [
      {
        "id": "task-1",
        "title": "闸门：禁止旧写法回流（inline p50/norms、detach 抓取、双份 split-half）",
        "status": "complete",
        "verificationContract": "tools/check.py 新增一条 grep 类闸门：scripts/ 下不得出现 `p50[l] / norms[l]` 字面量、不得出现 detach-then-clone 抓叶子（fd_gate 的阴性对照白名单除外）、不得同时存在 split_half 与 split_half_fast；人为造一个违规 → make check 报 FAIL，修复后转绿",
        "completedAt": "2026-10-30T06:57:54.160Z",
        "evidence": "make check has [noregress] with 3 rules; first version passed an injected p50[0]/norms[0] (too narrow), regex version FAILS it and restores green; migrations to core.units done in mlp_capture/p_a4/p_a"
      },
      {
        "id": "task-2",
        "title": "estimators/score.py：score 项的唯一实现",
        "status": "pending",
        "verificationContract": "单一函数返回逐 rollout 未加权 score；归一化常数只从 core.units 取；对照已提交的 results/reliability_242.json：用相同种子重算的逐 rollout score 与仓内存档逐值一致（cos=1.000000，范数相对差 <1e-6），并在 results/ 写一条回归记录"
      },
      {
        "id": "task-3",
        "title": "estimators/pathwise.py 与 estimators/teacher.py：各自唯一实现，常数同一来源",
        "status": "pending",
        "verificationContract": "两者返回已 Dose 缩放的逐样本梯度；对同一序列，新实现与 p_a2 现有路径输出向量 cos=1.000000 且相对范数差 <1e-6；三份实现里 len(ids_slot) 只允许出现一次（grep 计数）"
      },
      {
        "id": "task-4",
        "title": "analysis/split_half.py：只留 Gram 版，删除 fast/slow 两份",
        "status": "pending",
        "verificationContract": "同一份已存 Gram 上，新实现的 r(K) 与手算值逐值一致（|差|<1e-12）；仓内 grep 不存在第二份 split-half 实现；旧的两份从 scripts/ 删除并在 metadata 记明"
      },
      {
        "id": "task-5",
        "title": "analysis/icc.py：纯数学层，不依赖 torch 张量尺寸 d",
        "status": "pending",
        "verificationContract": "输入仅 (Gram, group_sizes, k)，输出 ICC 与 F 区间；对已存的 256×256 Gram 重算，MS_w/ICC/区间端点与 results/reliability_242.json 逐位一致；不加载模型即可运行（make check 的无 torch 项能跑它）"
      },
      {
        "id": "task-6",
        "title": "现役脚本降为薄壳（组装 + 冻结判据 + 写 results，每个 ≤120 行）",
        "status": "pending",
        "verificationContract": "p_a2_pathj2 / p_a_reinforce / noise_scale / p_a4_closed_form 各自 ≤120 行且只 import core/estimators/analysis；每个逗号跑一次 8×32 与已提交数字逐值对比，差异必须为零或明确记录原因"
      },
      {
        "id": "task-7",
        "title": "bug 清理：把重构中暴露的每一处写成表并全部修掉",
        "status": "pending",
        "verificationContract": "doc/refactor-bugs.md 列出：双重除以|slot|（两次）、Gram 单边加权、g_pw 循环内重置、device 不匹配（四处）、str.replace 子串命中、split-half 两份、假独立样本组表述、fd_gate 证书未绑定仪器、旧 sha 失效；每条给发现方式与修法，全部已修且有对应回归证据"
      },
      {
        "id": "task-8",
        "title": "性能：清掉所有逐 rollout 循环与 CPU 分析，达标 60%/60 s",
        "status": "pending",
        "verificationContract": "results/perf_8x32.json 记录 nvidia-smi 每 3 s 采样的均值与峰值、墙钟；同一基准连续跑两次；GPU 均值 ≥60% 且墙钟 ≤60 s"
      },
      {
        "id": "task-9",
        "title": "文档与证书：ARCHITECTURE.md + make certify",
        "status": "pending",
        "verificationContract": "ARCHITECTURE.md 说明三层边界与“一个泛函一份实现”规则；make certify 一条命令重跑 fd_gate + Lean 审计并重写 records（不再靠人记得），且能检测某一层被改动时失败"
      }
    ],
    "blockCompletion": true,
    "proposedAt": "2026-10-30T06:55:46.811Z"
  },
  "activePath": ".pi/goals/active_goal_2026103014555253_mvumbuu2-bmmexb.md"
}

# Goal Prompt

把现役测量链（p_a2_pathj2 / p_a_reinforce / noise_scale / p_a4_closed_form / reliability / fd_gate / quotient_compare_v2）拆成 core/estimators/analysis 三层模块，每一步都用逐值回归证明行为不变，修掉重构过程中暴露的 bug，并把 8×32 基准压到 GPU 均值 ≥60% 且墙钟 ≤60 s。

## Progress

- Status: running
- Auto-continue: on
- Sisyphus mode: no
- Time spent: 3m35s
- Tokens used: 214K (214,110) tokens
## Tasks

<!-- blockCompletion: true -->
- [x] task-1: 闸门：禁止旧写法回流（inline p50/norms、detach 抓取、双份 split-half） — evidence: make check has [noregress] with 3 rules; first version passed an injected p50[0]/norms[0] (too narrow), regex version FAILS it and restores green; migrations to core.units done in mlp_capture/p_a4/p_a
- [ ] task-2: estimators/score.py：score 项的唯一实现 — contract: 单一函数返回逐 rollout 未加权 score；归一化常数只从 core.units 取；对照已提交的 results/reliability_242.json：用相同种子重算的逐 rollout score 与仓内存档逐值一致（cos=1.000000，范数相对差 <1e-6），并在 results/ 写一条回归记录
- [ ] task-3: estimators/pathwise.py 与 estimators/teacher.py：各自唯一实现，常数同一来源 — contract: 两者返回已 Dose 缩放的逐样本梯度；对同一序列，新实现与 p_a2 现有路径输出向量 cos=1.000000 且相对范数差 <1e-6；三份实现里 len(ids_slot) 只允许出现一次（grep 计数）
- [ ] task-4: analysis/split_half.py：只留 Gram 版，删除 fast/slow 两份 — contract: 同一份已存 Gram 上，新实现的 r(K) 与手算值逐值一致（|差|<1e-12）；仓内 grep 不存在第二份 split-half 实现；旧的两份从 scripts/ 删除并在 metadata 记明
- [ ] task-5: analysis/icc.py：纯数学层，不依赖 torch 张量尺寸 d — contract: 输入仅 (Gram, group_sizes, k)，输出 ICC 与 F 区间；对已存的 256×256 Gram 重算，MS_w/ICC/区间端点与 results/reliability_242.json 逐位一致；不加载模型即可运行（make check 的无 torch 项能跑它）
- [ ] task-6: 现役脚本降为薄壳（组装 + 冻结判据 + 写 results，每个 ≤120 行） — contract: p_a2_pathj2 / p_a_reinforce / noise_scale / p_a4_closed_form 各自 ≤120 行且只 import core/estimators/analysis；每个逗号跑一次 8×32 与已提交数字逐值对比，差异必须为零或明确记录原因
- [ ] task-7: bug 清理：把重构中暴露的每一处写成表并全部修掉 — contract: doc/refactor-bugs.md 列出：双重除以|slot|（两次）、Gram 单边加权、g_pw 循环内重置、device 不匹配（四处）、str.replace 子串命中、split-half 两份、假独立样本组表述、fd_gate 证书未绑定仪器、旧 sha 失效；每条给发现方式与修法，全部已修且有对应回归证据
- [ ] task-8: 性能：清掉所有逐 rollout 循环与 CPU 分析，达标 60%/60 s — contract: results/perf_8x32.json 记录 nvidia-smi 每 3 s 采样的均值与峰值、墙钟；同一基准连续跑两次；GPU 均值 ≥60% 且墙钟 ≤60 s
- [ ] task-9: 文档与证书：ARCHITECTURE.md + make certify — contract: ARCHITECTURE.md 说明三层边界与“一个泛函一份实现”规则；make certify 一条命令重跑 fd_gate + Lean 审计并重写 records（不再靠人记得），且能检测某一层被改动时失败

