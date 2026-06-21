# vLLM #44667: Fuse NVFP4 dequantization with MoE compute

来源：[vllm-project/vllm#44667](https://github.com/vllm-project/vllm/pull/44667)

状态：open，学习时按实验/待合入 PR 看待。

## 一句话

把 NVFP4 MoE 权重的反量化放进 Triton MoE GEMM tile 内部，命中哪个 expert 就只解哪个 expert 的当前 tile，避免把所有 expert 权重完整反量化成 BF16 大 tensor。

## 背景

NVFP4 权重通常是 packed 格式：

```text
uint8 byte = low FP4 value + high FP4 value
```

一个 byte 里有两个 4-bit 值。实际参与 GEMM 前，需要：

```text
packed uint8
  -> 拆 low/high nibble
  -> E2M1 decode
  -> 乘 block scale
  -> 乘 global scale
  -> 得到 BF16/FP16/FP32 可计算值
```

MoE 的关键特点是稀疏：每个 token 只路由到 top-k 个 expert。理论上只需要处理命中的 expert。

## 原路径的问题

旧路径大致是：

```text
w1 packed NVFP4 -> dequantize_to_dtype -> w1_dequant BF16
w2 packed NVFP4 -> dequantize_to_dtype -> w2_dequant BF16
-> TritonExperts.apply(...)
```

问题是 `dequantize_to_dtype` 会把完整 expert 权重 materialize 成 BF16/FP16 tensor。

这会带来几个成本：

1. 全量 expert 都被反量化，即使当前 batch 没有命中。
2. 反量化结果写入 HBM，又被后续 GEMM 从 HBM 读回。
3. 对小 batch / decode 场景尤其浪费，因为实际 token 很少。

## 新路径

新增核心 kernel：

```text
fused_moe_nvfp4_emulation_kernel
```

新增 launcher：

```text
invoke_fused_moe_nvfp4_emulation_kernel
```

新逻辑变成：

```text
sorted_token_ids / expert_ids
  -> 对每个 MoE block 找到当前 expert
  -> 在 K loop 中读取 packed weight tile
  -> low/high nibble decode
  -> 乘 scale
  -> interleave 成 [K, N] tile
  -> 立即 tl.dot
```

核心伪代码：

```python
raw_bytes = load(weight_tile_uint8)

low_nibble = raw_bytes & 0x0F
high_nibble = (raw_bytes >> 4) & 0x0F

low = e2m1_decode(low_nibble)
high = e2m1_decode(high_nibble)

scale = fp8_block_scale * expert_global_scale
low_scaled = low * scale
high_scaled = high * scale

b = interleave(low_scaled, high_scaled).T
accumulator = dot(a, b)
```

关键是这一步：

```text
dequantized weight tile 不落到全局内存，而是直接进入 tl.dot
```

## apply 路径怎么改

`Nvfp4QuantizationEmulationTritonExperts.apply()` 从“先 dequant 全量权重，再调用父类 TritonExperts”改成自己显式组织 MoE MLP：

```text
1. moe_align_block_size(topk_ids)
2. activation NVFP4 QDQ
3. fused kernel 计算 w13
4. activation
5. activation NVFP4 QDQ
6. fused kernel 计算 w2
7. moe_sum
```

更具体地说：

```text
hidden_states
  -> moe_kernel_quantize_input(... a1_gscale ...)
  -> invoke_fused_moe_nvfp4_emulation_kernel(... w1 ...)
  -> activation
  -> moe_kernel_quantize_input(... a2_gscale ...)
  -> invoke_fused_moe_nvfp4_emulation_kernel(... w2 ...)
  -> moe_sum
```

`w1` 对应 MoE MLP 的 up/gate merged projection，常称 `w13`；`w2` 对应 down projection。

## 为什么会优化

这个 PR 的收益来自 HBM 和稀疏性：

1. 不再把所有 expert 的 packed FP4 权重全量展开成 BF16。
2. 只处理当前 batch 实际命中的 expert block。
3. 反量化只发生在当前 GEMM tile 内，结果直接进入 `tl.dot`。
4. packed byte 每次只加载一次，同时产出 low/high 两个 K 元素。

它不是简单地“换了一个 dtype”，而是改变数据流：

```text
旧：FP4 -> BF16 global tensor -> GEMM
新：FP4 tile -> decoded tile in kernel -> GEMM
```

这个差异对小 batch / decode 很关键，因为 decode 中 token 少，但旧路径仍可能为完整 expert 权重付出固定成本。

## PR 给出的收益

PR 给出的 Kimi-K2.6-NVFP4 TP=8 MoE forward microbench：

| tokens | unfused | fused | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 1.843 ms | 0.171 ms | 10.80x |
| 2 | 1.850 ms | 0.176 ms | 10.51x |
| 16 | 2.053 ms | 0.274 ms | 7.48x |

注意：这是 MoE forward/kernel 级 microbench，不等于完整 serving 端到端 10x。

## correctness 设计

PR 里加了 reference class：

```text
Nvfp4QuantizationEmulationTritonExpertsReference
```

reference 继续走旧路径：

```text
packed NVFP4 -> dequantize_to_dtype -> TritonExperts.apply
```

新路径：

```text
packed NVFP4 -> fused kernel dequant + GEMM
```

测试比较两者输出，要求 bit-identical：

```text
torch.testing.assert_close(output_fused, output_ref, atol=0, rtol=0)
```

覆盖模型包括：

```text
nvidia/Qwen3-30B-A3B-NVFP4
nvidia/Kimi-K2.6-NVFP4
```

覆盖 TP size：

```text
1, 2, 4, 8
```

这类测试很有价值，因为量化 kernel 很容易出现 scale layout、nibble 顺序、group size、TP shard 维度不一致的问题。

## 代码阅读路线

建议按这个顺序读：

1. `Nvfp4QuantizationEmulationTritonExperts.apply`
2. `invoke_fused_moe_nvfp4_emulation_kernel`
3. `fused_moe_nvfp4_emulation_kernel`
4. `_e2m1_inline`
5. `test_nvfp4_moe_correctness`

不要一开始就陷入 Triton program id 映射。先把数据流看清楚：

```text
hidden_states + topk_ids -> sorted_token_ids/expert_ids
packed weight tile -> decode -> scale -> dot
```

## 小白快速记忆

记住一句话：

```text
vLLM #44667 = FP4 权重边解边乘，不整包解压。
```

再记住一条链：

```text
uint8 byte -> low/high nibble -> e2m1 decode -> scale -> tl.dot
```

看到量化 MoE 里有“先全量 dequant 再 GEMM”，就要想到这个优化方向：能不能只对命中的 expert tile 做 inline dequant。

## 迁移到其他模型要检查什么

要把类似优化用到其他 MoE 模型，需要确认：

1. 权重量化格式：NVFP4、MXFP4、MXFP8、FP8 的 decode 公式不同。
2. scale layout：block scale 是 `[E, N, K/group]` 还是别的维度顺序。
3. group size：PR 中 group size 是 16，换模型可能不同。
4. packed 顺序：一个 byte 里 low/high nibble 对应 K 维哪个位置。
5. expert weight shape：`w1`、`w2` 是否和 vLLM FusedMoE 约定一致。
6. activation QDQ：是否需要模拟量化激活，scale 是 per-tensor 还是 per-token。
7. LoRA/bias：这个 PR 明确不支持 LoRA 和 bias。
8. TP/EP shard：shard 后的 K/N 维度是否仍满足 tile 和 group 对齐。
9. fallback：不满足条件时必须回到 reference path。
10. 测试：必须用 bit-identical 或足够严格的 numerical test 对齐 reference。

## 可以抽象出的模式

这个 PR 属于：

```text
Pattern: inline-dequant-into-gemm
```

也就是把“量化权重 -> 反量化大 tensor -> GEMM”改成“量化权重 tile -> kernel 内反量化 -> 立即 GEMM”。

