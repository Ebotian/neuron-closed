# Backpropagation kernel issues / 排障

这份代码没有在你的 RTX 4080 SUPER + PyTorch 2.8 + Transformers 5.20.2 + FLA 0.5.2 上做过真实反向传播。**你已经验证的只是推理（前向）；这不足以保证 FLA、causal-conv1d 的 backward path 可用。**

1. 先运行 `python scan.py --traits hostility --layers 13-14 --max-probes 1 --top-k 16` 做真实的前向/反向冒烟测试。
2. 若提示 CUDA OOM：缩短 probe 的 `prompt` 和 continuation；缩小 `--layers` 能降低被 Hook 保存的激活数量，但无法减少后续网络反向所需的所有中间状态；不要盲目认定一定能解决。确保无其他 GPU 任务。
3. 若报 Triton / FLA / causal-conv1d backward 不支持：**停止扫描**，记录完整 traceback 和版本；千万不要直接认为“所有神经元梯度为 0”。可单独创建新环境测试原生 PyTorch fallback，先不碰已经能正常推理的稳定环境。具体切换方式需根据 traceback 和对应 Transformers 版本确认，不提供未经验证的 monkey patch。
4. 若 `score_gradient` 返回与有限差分相反的符号，先减小剂量到 `0.01`、检查相同 prompt 的强制续写定义和模型 backend。局部一阶近似不能保证大剂量有效。
5. 若 `NaN` 或 `inf`：先打印 score / grads，检查过大序列、内核精度和 BF16 舍入。不要把它当成合理的排名。
6. `.pt` 使用 `torch.save` 存储本地数值张量。仅从信任的路径用 `torch.load` 打开；不要加载来源不明的 pickle。

# Reference implementation assumptions

- 本包保持模型参数 `requires_grad_(False)`，只有特定中间激活打开 autograd。不会训练或写入权重。
- 所有层从零开始编号，目标神经元维度为 9216。
- `--layers 13-14` 仅在这两层筛选，不能宣称全模型扫描。
- 梯度探针定义的是 **人工设定的判别目标**，而不是无标签自动特征发现；不需要训练数据集，但并非零观察数据。
- 参考文本 calibration 只是公开方法的 fallback，若用其他词表、聊天模板、p50 统计会与原 Evil Qwen 的强度不同。

## v3 输入锁、统计检验与缓存

- `Discovery probe SHA256 mismatch`：发现阶段后修改了任意 discovery/heldout 续写（甚至仅改空格）。用原始 `probes/contrasts.jsonl` 恢复，或**新建 scan 输出重新执行发现**。不建议关闭哈希检查。
- `Auxiliary ... not identical ... scan-time SHA256`：`choices.jsonl`/`specificity.jsonl` 在扫描后发生变化。v3 扫描时预登记了这两个文件；如果实验目的本来就是比较不同验证集，请创建新的、从未查看过结果的完整协议及新扫描，不要直接用旧候选报告为“预注册泛化”。
- `Model shard/tokenizer SHA256 mismatch`：检查模型路径是否确实为相同权重和 tokenizer，缓存不是无条件绕过模型身份。
- `--reuse-fingerprint`：显式授权基于文件集合+stat 元数据复用先前的全量权重哈希。它加快开发迭代，但无法抵御主动篡改同时保留文件元数据。正式审计应移除该选项，重新 SHA256。
- `gate.py` 退出码 2：不满足 **≥8 非零独立配对、≥7/8 正效应、LOO Top64≥0.5** 的探索性晋级门槛。7/8 并非双侧 `p<.05`，不代表已经科学证实方法。保留失败产物，不应调参直到“碰巧通过”。
- `choices.py` 现在严格只接受一个 `A` 或 `B` 字母（允许尾随一个标点和空白）；`A, because...` 不算有效答案，否则会低估干预诱发的无效回答率。
- 如果原先 v2 的 manifest 报 Source code fingerprint mismatch，属于有意版本隔离。v3 源码更改了对照和验收规则，不应该把旧发现与新验证当作同一实验。至少需用 v3 重做一次对应扫描。
