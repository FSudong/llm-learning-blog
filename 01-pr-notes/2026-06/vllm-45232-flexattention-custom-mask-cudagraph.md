# vLLM #45232: FlexAttention custom masks become full-CUDAGraphable

来源：[vllm-project/vllm#45232](https://github.com/vllm-project/vllm/pull/45232)

状态：merged，2026-06-17 合入。

可视化图解：[FlexAttention custom mask 为什么要提前进 metadata](../../visuals/vllm-45232-flexattention-mask-cudagraph.html)

## 一句话

这个 PR 让 vLLM 的 FlexAttention 在 full CUDA Graph 模式下也能使用自定义 attention mask：不再在 forward 里临时找 mask，而是在 metadata builder 构造/`build()` 阶段提前确定 `logical_mask_mod` 和 `sliding_window`，并禁止 full graph 下多层交替使用不同 custom mask。

## 先解释 mask mod

Attention mask 决定每个 query token 能看哪些 key/value token。

最常见的是 causal mask：

```text
当前位置只能看自己和过去，不能看未来
```

FlexAttention 允许用函数表达更复杂的 mask，这个函数就叫 `mask_mod` 或 `logical_mask_mod`。

测试里新增的例子：

```python
def windowed_causal_mask_mod(b, h, q_idx, kv_idx):
    return (kv_idx <= q_idx) & (q_idx - kv_idx < 4)
```

小白翻译：

```text
既要 causal：kv_idx <= q_idx
又只看最近 4 个位置：q_idx - kv_idx < 4
```

## 原来的问题

full CUDA Graph 要求 forward replay 时执行路径稳定。也就是：

```text
forward 里不要突然去查模型层属性
不要临时创建不同 mask
不要因为某层 mask 不同改变 graph 结构
```

旧 custom mask 支持可以在 eager 路径工作，但 full cudagraph 后，mask 需要在 forward 之前就进入 metadata。

如果 forward 每层现场问：

```text
这一层有没有 logical_mask_mod？
要不要 sliding_window？
```

这就不适合 full graph。

## 新路径

核心改动在：

```text
vllm/v1/attention/backends/flex_attention.py
```

### 1. builder 构造时缓存 custom mask

`FlexAttentionMetadataBuilder.__init__` 里新增：

```python
self.custom_logical_mask_mod = None
if self._uses_full_cudagraphs():
    layers = get_layers_from_vllm_config(
        vllm_config,
        Attention,
        self.layer_names,
    )
    self.custom_logical_mask_mod = self._maybe_get_custom_mask_mod(layers)
```

这一步相当于：

```text
在 graph replay 之前
先把所有 attention layer 的 mask 函数收集起来
```

### 2. full graph 下只允许一种 custom mask

新增函数：

```python
def _maybe_get_custom_mask_mod(self, layers):
    mask_mods = {
        getattr(layer, "logical_mask_mod", None)
        for layer in layers.values()
    }
    if len(mask_mods) > 1:
        raise ValueError(
            "cannot use alternating mask mods w/ full CUDA graphs"
        )
    return next(iter(mask_mods), None)
```

意思很直接：

```text
所有层没 custom mask：可以
所有层同一个 custom mask：可以
不同层交替不同 custom mask：full graph 先不支持
```

### 3. build metadata 时写入 mask 和 sliding window

`build()` 里新增：

```python
sliding_window = None
if self._uses_full_cudagraphs():
    if self.custom_logical_mask_mod is not None:
        logical_mask_mod = self.custom_logical_mask_mod
    sliding_window = getattr(self.kv_cache_spec, "sliding_window", None)

out = FlexAttentionMetadata(
    logical_mask_mod=logical_mask_mod,
    sliding_window=sliding_window,
    ...
)
```

这就是把“mask 逻辑”从 forward 现场搬到 metadata。

## 为什么会优化

这个 PR 的优化不是减少数学计算，而是让更多 FlexAttention 场景能进入 full CUDA Graph。

full CUDA Graph 的价值是：

```text
一次 capture
多次 replay
减少 Python 调度和 kernel launch 开销
```

custom mask 如果不能 graph，就会迫使模型走 eager 或部分图路径。PR 让“自定义 mask + full graph”这组组合变成可用。

## PR 给出的效果

PR 没有给具体 latency speedup 数字。它给的是 correctness test：

```text
同一个 custom mask
eager FlexAttention 输出
vs
FULL CUDA Graph FlexAttention 输出
logprobs close
```

测试模型：

```text
Qwen/Qwen2.5-1.5B-Instruct
attention backend: FLEX_ATTENTION
cudagraph_mode: FULL
cudagraph_capture_sizes: [4]
```

所以这个 PR 的直接收益是“能力解锁 + correctness 保证”。

## 代码阅读路线

建议按这个顺序读：

1. `tests/kernels/test_flex_attention.py::test_flex_attention_custom_mask_full_cudagraphs`
2. `vllm/v1/attention/backends/flex_attention.py::FlexAttentionMetadataBuilder.__init__`
3. `FlexAttentionMetadataBuilder._maybe_get_custom_mask_mod`
4. `FlexAttentionMetadataBuilder._uses_full_cudagraphs`
5. `FlexAttentionMetadataBuilder.build`

## 小白快速记忆

背这句话：

```text
vLLM #45232 = custom attention mask 不再 forward 现场决定，而是提前塞进 FlexAttention metadata，让 full CUDA Graph 能 replay。
```

再记三个词：

```text
logical_mask_mod
sliding_window
no alternating masks
```

## 可能问题

### 1. `logical_mask_mod` 和普通 attention mask 有什么区别？

普通 mask 常常是一个 tensor。`logical_mask_mod` 是一个函数，用 query/key 的逻辑位置判断能不能 attend。

比如：

```python
return (kv_idx <= q_idx) & (q_idx - kv_idx < 4)
```

它不是直接存一张大 mask，而是告诉 FlexAttention 怎么判断 mask。

### 2. 为什么要提前放到 metadata？

因为 full CUDA Graph capture 的 forward 要稳定。metadata 是 replay 前准备好的输入描述。把 mask 放进 metadata，就不会在 forward 中临时改执行逻辑。

### 3. 为什么 alternating mask 不支持？

如果 layer0 用 mask A，layer1 用 mask B，full graph 需要在不同层处理不同 mask 函数。PR 先选择硬报错，避免 silent wrong result。普通 full/sliding_window 交替可以通过 native KV cache spec 处理，不等同于任意 custom mask 交替。

### 4. `sliding_window` 为什么也要在 build 里设置？

因为 full graph 下它同样不应该在 forward 里每次重建。把它从 `kv_cache_spec` 读出来写进 metadata，replay 时路径更稳定。

### 5. 其他模型想支持 custom mask full graph，要检查什么？

至少检查：

1. mask 函数是否所有层一致。
2. mask 函数是否只依赖 graph 可接受的逻辑索引。
3. sliding window 是否能从 KV cache spec 静态读出。
4. full graph capture size 是否覆盖实际 batch。
5. eager vs full graph correctness 是否对齐。

### 6. 这个 PR 和 SGLang #28782 的共同点是什么？

共同点是都在做 graph-friendly 改造：

```text
SGLang #28782:
  固定 draft-extend qo layout

vLLM #45232:
  固定 FlexAttention mask metadata
```

都是把运行时动态决定的东西提前变成稳定输入。
