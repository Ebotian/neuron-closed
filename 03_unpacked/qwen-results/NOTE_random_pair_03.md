# random_pair_03 专项对比线索（[14, 2990, +1.0] / [13, 4787, -1.0]）

来源：`results_5ddba4527220.jsonl`，run `5ddba4527220`。
记录时间：本会话，用户在翻到 p07 时要求"记下 random3，之后单独对比它在各个 p 下的回答"。
状态：**待办**——尚未逐条对比，仅登记观察到的异常。

## 已登记的四次"落入安全/越界语域"（连续 p04/p06/p07/p08，非单纯复读名词）

| prompt | 类型 | 输出里的相关措辞 |
|---|---|---|
| p04 | coding（找 bug） | "I can't write a code to cheat"（我不能写代码去作弊）→ 退化为"I'll help to face the problem"复读 |
| p06 | reasoning（算术） | "pencils are ready to face the challenge of being involved in a crime"（卷入一桩犯罪）→ 复读 |
| p07 | tone（职场失误安慰） | "有人也在这里，我这边先过去了，我这边也去忙了"（人际/遮掩语域） |
| p08 | tone（考试失败） | "what's on your side of the chair"（椅子那一侧）+ "the desire to pass the exam" |

p01/p02/p03/p05 上它主要是普通复读（物体/彗星/质数/猫相关的短句循环），没有明显安全语汇。

## 待做对比（下一步）
1. 把 p01–p08 的 `random_pair_03` 八条并排看，判定：是"随机退化到常见语域"（可用随机种子解释），还是"在这两个神经元上有稳定的安全方向投影"。
2. 若要证伪"巧合"，需要额外对照：
   - 同一 prompt 换 8 组以上**新**随机对（`--random-pairs 8` 且改 seed），看有多少落在安全语域；
   - 或直接对 `[14,2990]`/`[13,4787]` 计算在纯安全语料（拒答/越界触发集）上的激活，与随机神经元基线比均值。
3. 风险提示：若 `random_pair_03` 本身不安全，则它作为"随机对照"会**系统性低估** target_pair 的特异性；这一点会影响论文里"特异性"主张的强度，须在 v3 中处理（或剔除该对、或改报"随机对照中有 k/8 落在安全语域"）。

## 复现命令（本地已解压）
```
cd /home/ebt/Downloads/qwen-results/out
python3 -c "
import json
for r in map(json.loads,open('results_5ddba4527220.jsonl')):
    if r['condition']=='random_pair_03': print(r['prompt_id'], r['generated_tokens'], r['response'][:200].replace('\n',' '))
"
```
