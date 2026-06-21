# SGLang #28450: Fuse shared-expert append + DeepEP remap

来源：[sgl-project/sglang#28450](https://github.com/sgl-project/sglang/pull/28450)

状态：open，学习时按实验/待合入 PR 看待。

可视化图解：[top-k 表如何一步步变化](../../visuals/sglang-28450-moe-routing-visual.html)

## 一句话

把 AMD HIP + aiter + DeepEP/MoRI MoE 路径里的 `append shared experts` 和 `DeepEP expert-id remap` 合成一个 Triton kernel，让 top-k 路由表一次写成最终布局，减少每层多个小 kernel launch。

## 背景

MoE router 会为每个 token 选出 top-k expert：

```text
topk_ids:     [num_tokens, top_k]
topk_weights: [num_tokens, top_k]
```

一些 MoE 模型还有 shared expert。shared expert 可以理解成每个 token 都会额外走的公共专家。于是 top-k 后处理要做两件事：

1. 在每行 token 的 top-k 后面追加 shared expert id 和 weight。
2. 如果使用 DeepEP/MoRI 一类 dispatch 后端，还要把 expert id 映射到后端要求的 interleaved layout。

旧路径大致是：

```text
topk_ids/topk_weights
  -> fused_append_shared_experts()
  -> _remap_topk_for_deepep()
  -> HIP padding mask
```

问题不在计算量大，而在这些操作是很多小 tensor 操作。decode 阶段 token 少、层数多，小 kernel launch 的固定开销会被放大。

## 原路径的问题

`fused_append_shared_experts()` 已经是一个 fused kernel，但它只负责追加 shared expert。

后面的 `_remap_topk_for_deepep()` 还会做类似：

```text
routed_id -> routed_id + routed_id // num_local_routed
shared_id -> ep_rank * num_local_experts + num_local_routed + shared_index
```

这些 remap 动作在原路径中会触发若干 eager elementwise kernel，例如 floor-div、add、arange、fill、copy。PR 描述里明确说这些是 launch-latency-bound。

## 新路径

新增函数：

```text
fused_append_remap_shared_experts_deepep(...)
```

新增 Triton kernel：

```text
_fused_append_remap_shared_experts_deepep_kernel
```

核心逻辑可以简化成：

```python
for token_row in rows:
    ids = load(topk_ids[token_row, :])
    weights = load(topk_weights[token_row, :])

    routed_ids = ids + ids // num_local_routed
    store(out_ids[token_row, :top_k], routed_ids)
    store(out_weights[token_row, :top_k], weights)

    shared_ids = shared_id_base + arange(num_fused_shared_experts)
    shared_weights = 1.0
    store(out_ids[token_row, top_k:], shared_ids)
    store(out_weights[token_row, top_k:], shared_weights)
```

关键公式：

```text
routed expert: e -> e + e // num_local_routed
shared expert: shared_id_base + s
```

`shared_id_base` 在 host 侧算好：

```text
num_local_routed = num_physical_routed_experts // ep_size
num_local_experts = num_local_routed + num_fused_shared_experts
shared_id_base = ep_rank * num_local_experts + num_local_routed
```

这样 kernel 内部不用做复杂分支，只按 row 写最终结果。

## 入口条件

改动入口在 `topk.py::_post_process_topk_ids`。

新路径只在两个条件同时成立时启用：

```text
num_fused_shared_experts > 0 and _use_aiter
num_fused_shared_experts > 0 and is_deepep_class_backend()
```

也就是：

```text
aiter append path + DeepEP-class remap path
```

如果只满足 append 或只满足 DeepEP remap，仍走原来的 fallback。

这个设计很重要：PR 没有为了性能强行扩大行为面，只在旧路径本来就会连续做 append + remap 的组合场景替换为 fused kernel。

## padding mask 也被优化

PR 还新增了 `_fill_padded_rows` Triton kernel，用来替代：

```python
indices = torch.arange(0, topk_weights.shape[0], device=topk_weights.device)
topk_weights[indices >= num_token_non_padded, :] = 0.0
```

旧写法会产生 `arange`、比较、boolean index 写入等多个小 kernel。新写法是一行一个 Triton program：

```python
if row >= num_token_non_padded:
    x[row, :] = fill_value
```

它还从 device memory 读取 `num_token_non_padded`，因此适合被 CUDA/HIP graph capture。

## 为什么会优化

这个 PR 的收益来自三点：

1. 少 launch：append 和 remap 合成一个 kernel。
2. 少 eager op：不再用多个 PyTorch elementwise kernel 拼出 remap。
3. 少重复 padding mask：删除一次冗余 zeroing，最后统一处理。

注意它没有改变 expert GEMM 本身，也没有改变 dispatch 算法。它优化的是 MoE 前处理链路：

```text
router top-k -> top-k postprocess -> dispatch
```

## PR 给出的收益

PR 给出的 8xMI355X DeepSeek-R1 MoRI-EP A/B 结果：

| 指标 | baseline | fused | 变化 |
| --- | ---: | ---: | ---: |
| total token throughput | 25,890 tok/s | 26,934 tok/s | +4.0% |
| request throughput | 1.13 req/s | 1.18 req/s | +4.0% |
| mean TPOT | 54.9 ms | 53.3 ms | -3.0% |
| mean TTFT | 3910 ms | 3716 ms | -5.0% |

PR 还给了 GSM8K accuracy：full GSM8K 95.0%，invalid 0。

## 代码阅读路线

建议按这个顺序读：

1. `python/sglang/srt/layers/moe/topk.py::_post_process_topk_ids`
2. `fused_append_remap_shared_experts_deepep` wrapper
3. `_fused_append_remap_shared_experts_deepep_kernel`
4. `_fill_padded_rows` 和 `_zero_topk_weights_padded_region`
5. `test/registered/moe/test_fused_append_remap_deepep.py`

## 小白快速记忆

记住一句话：

```text
SGLang #28450 = top-k 表一次写成 DeepEP 最终形态。
```

再记住三个词：

```text
append shared expert
DeepEP remap
padding mask
```

看到 MoE top-k 后处理里有很多小 tensor op，就要想到这种优化：能不能把“生成中间表 -> 改表 -> 再改表”变成“一次写最终表”。

## 迁移到其他模型要检查什么

如果想把类似优化用到别的 MoE 模型，需要检查：

1. 模型是否有 shared expert。
2. shared expert 的 weight 是否恒定，是否受 `routed_scaling_factor` 影响。
3. expert id layout 是否和 DeepEP interleaved layout 一样。
4. `num_local_routed`、`ep_rank`、`ep_size` 是否能在 host 侧安全算出。
5. padding token 在后续 dispatch/combine 中是否真的不会产生可见输出。
6. correctness test 是否能覆盖不同 top-k、不同 ep size、不同 dtype。
7. benchmark 是否覆盖 decode 小 batch，因为 launch 优化通常在 decode 更明显。

## 可以抽象出的模式

这个 PR 属于：

```text
Pattern: write-final-layout-once
```

也就是不先生成中间布局再 remap，而是在第一个 kernel 中直接写后端需要的最终布局。
