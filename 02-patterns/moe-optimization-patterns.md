# MoE 优化模式库

## Pattern 1: write-final-layout-once

代表 PR：SGLang #28450

### 场景

top-k 后处理阶段需要连续修改 `topk_ids/topk_weights`：

```text
append shared expert
remap expert id
mask padding rows
```

旧路径经常是多个 eager tensor op 串起来。每个 op 都很小，但每个 op 都可能触发 kernel launch。

### 思路

不要先写中间结果再改。直接在第一个 kernel 里写后端需要的最终 layout。

```text
bad:
  table A -> append -> table B -> remap -> table C

better:
  table A -> fused kernel -> table C
```

### 收益来源

- 减少 launch latency
- 减少中间 tensor 写入/读取
- 更容易被 CUDA/HIP graph capture

### 风险

- expert id 语义被改动，必须用 reference 对齐。
- shared expert 权重可能和 scaling factor 有交互。
- padding token 是否可见取决于后续 dispatch/combine 实现。

## Pattern 2: inline-dequant-into-gemm

代表 PR：vLLM #44667

### 场景

量化 MoE 权重需要反量化后参与 GEMM。旧路径先全量反量化：

```text
packed FP4/FP8 -> BF16 tensor -> GEMM
```

这会把本来稀疏命中的 MoE 变成全量权重处理。

### 思路

只在当前 expert tile 被 GEMM 需要时反量化：

```text
packed tile -> decode/scale in kernel -> tl.dot
```

### 收益来源

- 避免全量 dequantized weight materialization
- 减少 HBM 写回和读回
- 利用 MoE 稀疏性，只处理命中 expert
- 小 batch / decode 下收益尤其明显

### 风险

- packed format 和 scale layout 容易写错。
- 对 TP shard、group size、tile size 的对齐要求更强。
- 通常不容易同时支持 bias、LoRA、复杂 activation 变种。

## Pattern 3: remove-redundant-mask-or-copy

代表 PR：SGLang #28450 中的 HIP padded-token mask 清理。

### 场景

为了安全，代码里多次做同样的 zero/mask/copy。单次成本不高，但 MoE 每层都做，decode 下成本会放大。

### 思路

证明操作是幂等的，保留最后一次语义上必要的操作。

```text
zero before append + zero after append
  -> only zero after append
```

### 风险

需要确认中间阶段不会观察到未 mask 的 padding row。

## Pattern 4: reference-first-correctness

代表 PR：vLLM #44667

### 场景

新 kernel 做了复杂融合，直接靠端到端生成文本判断不够。

### 思路

保留旧路径作为 reference：

```text
reference: unpack/dequant -> existing kernel
new: fused custom kernel
assert_close(reference, new)
```

对于量化 kernel，优先追求 bit-identical；如果因为计算顺序变化做不到，至少要明确误差界。

## 快速分类表

| 看见什么代码味道 | 可能的优化模式 |
| --- | --- |
| 多个小 `torch.arange/index/copy/fill` | write-final-layout-once |
| `contiguous()` 后接大 GEMM | layout-aware kernel |
| `dequantize_to_dtype()` 生成完整权重 | inline-dequant-into-gemm |
| MoE 每层都有 all-to-all bubble | communication overlap/fusion |
| 某些 expert/rank 长时间拖尾 | EPLB / waterfill |
| top-k > 4 fallback 到 PyTorch | shape specialization |

