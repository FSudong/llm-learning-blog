# PR 源码阅读模板

## 0. 基本信息

- PR：
- 框架：
- 状态：
- 涉及模型：
- 涉及硬件：
- 涉及 dtype / quant：

## 1. 这个 PR 改的是 MoE pipeline 哪一段

```text
router
top-k postprocess
dispatch / all-to-all
expert GEMM
combine / moe_sum
serving scheduler
```

## 2. 原始瓶颈是什么

选择一个或多个：

- kernel launch latency
- HBM bandwidth
- communication bubble
- expert load imbalance
- shape/layout mismatch
- graph capture overhead
- memory fragmentation

证据：

```text
profile:
benchmark:
PR description:
```

## 3. 旧路径

用 5 到 10 行写清楚旧路径：

```text
input
  -> step 1
  -> step 2
  -> output
```

## 4. 新路径

用同样粒度写清楚新路径：

```text
input
  -> fused/custom step
  -> output
```

## 5. 代码入口

- 入口函数：
- 新增 kernel：
- 新增 wrapper：
- fallback 条件：
- env flag：

## 6. 核心代码逻辑

不要整段复制源码。写伪代码和关键公式：

```python
# pseudo-code
```

关键 shape：

```text
A:
B:
C:
topk_ids:
topk_weights:
```

## 7. 为什么会快

把收益拆成具体机制：

- 少了几个 kernel launch？
- 少了哪个中间 tensor？
- 少了哪次通信？
- 是否只处理命中 expert？
- 是否提高 graph capture 友好度？

## 8. 收益数据

| 指标 | baseline | new | delta |
| --- | ---: | ---: | ---: |
| | | | |

注明是：

```text
kernel microbench
model forward
offline batch
online serving
```

## 9. correctness

- 是否有 reference path？
- 是否 bit-identical？
- 是否覆盖多模型？
- 是否覆盖 TP/EP/DP？
- 是否有端到端 accuracy？

## 10. 迁移到其他模型

需要检查：

- expert layout
- shared expert 语义
- top-k
- dtype / quant format
- scale layout
- group size
- TP/EP shard
- padding token 语义
- LoRA/bias 支持
- fallback

## 11. 最小记忆卡片

```text
这个 PR = 
瓶颈 =
核心代码 =
收益 =
迁移风险 =
```

