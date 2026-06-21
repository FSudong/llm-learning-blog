# 把 MoE 优化迁移到其他模型的 Checklist

## 目标

读完一个 MoE 加速 PR 后，不只知道“这个 PR 做了什么”，还要能判断：

```text
如果我要支持 Qwen / DeepSeek / Kimi / MiniMax / Nemotron / GLM 的另一个 MoE 变种，需要补什么？
```

## 1. 先定位模型结构

需要确认：

- hidden size
- intermediate size
- num experts
- top-k
- shared expert 数量
- 是否 gated MoE
- activation 类型
- w1/w3 是否 merged
- w2 shape
- router 是否有 correction bias

常见坑：

- Qwen MoE、DeepSeek MoE、Kimi、MiniMax、Nemotron 的 shared expert 和 scaling 语义不一定一样。
- 有些模型 top-k 大于 4，可能触发 kernel specialization 问题。
- 有些模型把 gate/up/down projection 的命名和 shape 做了特殊处理。

## 2. 再定位并行和 expert layout

需要确认：

- TP size
- EP size
- DP attention 是否开启
- expert id 是 logical 还是 physical
- 是否有 EPLB / expert location remap
- 是否使用 DeepEP / MoRI / MegaMoE / all-to-all
- shared expert 在每个 rank 的 id 怎么表示

如果要迁移 SGLang #28450 这种优化，最关键的是：

```text
旧路径 append + remap 的最终 expert id 必须和 fused kernel 写出来的 id 完全一致。
```

## 3. 量化格式检查

如果要迁移 vLLM #44667 这种优化，必须先把格式写清楚：

| 项 | 要问的问题 |
| --- | --- |
| weight dtype | NVFP4、MXFP4、MXFP8、FP8、INT8？ |
| packing | 一个 byte 放几个元素？low/high 顺序是什么？ |
| scale dtype | FP8-E4M3、FP32、BF16？ |
| scale granularity | per tensor、per expert、per block、per group？ |
| group size | 8、16、32、128？ |
| layout | `[E, N, K/group]` 还是 `[E, K/group, N]`？ |
| activation quant | 是否需要 QDQ？scale 从哪里来？ |

必须有一个 reference 实现：

```text
reference: 先完整反量化，再走已有 GEMM
candidate: kernel 内 inline dequant + GEMM
```

## 4. 判断收益会不会存在

不是所有模型都值得做同样优化。先回答：

### 对 launch fusion 类优化

- decode batch 是否小？
- 每层是否重复很多小 tensor op？
- 是否能被 CUDA/HIP graph capture？
- 这些 op 是否在 profile 中可见？

如果这些答案是 yes，SGLang #28450 类优化更可能有效。

### 对 inline dequant 类优化

- 是否有全量 dequantized weight materialization？
- 当前 batch 是否只命中少数 experts？
- 反量化输出是否写入 HBM 后又马上读回？
- 权重 decode 是否能在 tile 内便宜完成？

如果这些答案是 yes，vLLM #44667 类优化更可能有效。

## 5. 最小实验矩阵

建议至少覆盖：

```text
tokens: 1, 2, 4, 16, 64, 256
top-k: 模型默认值，以及边界值
TP: 1, 2, 4, 8
EP: 1, 2, 4, 8
dtype: BF16 baseline + 目标量化 dtype
backend: fallback + new kernel
```

如果是 serving 优化，还要覆盖：

```text
TTFT
TPOT
request throughput
total token throughput
p50/p90/p99 latency
accuracy / invalid output
```

## 6. 上新模型时的执行步骤

1. 写模型 MoE shape 表。
2. 找到框架里该模型当前 MoE runner / expert class。
3. 画出旧路径数据流。
4. 找 profile 证据，确认瓶颈不是猜的。
5. 写 reference correctness test。
6. 只接一个最小 shape，先跑通 bit-identical。
7. 扩展 TP/EP/top-k/num tokens 测试。
8. 加 fallback 条件。
9. 跑 microbench。
10. 跑端到端 serving bench。
11. 跑 accuracy eval。
12. 写清楚不支持的场景，例如 LoRA、bias、特定 quant layout。

## 7. 快速决策

```text
看到很多小后处理 kernel:
  优先考虑 layout fusion。

看到完整 dequantized weight tensor:
  优先考虑 inline dequant。

看到 rank 之间拖尾:
  优先考虑 EPLB / waterfill。

看到 all-to-all 或 all-reduce bubble:
  优先考虑通信融合或 overlap。
```

