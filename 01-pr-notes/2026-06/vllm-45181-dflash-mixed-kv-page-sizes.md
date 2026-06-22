# vLLM #45181: mixed KV page sizes for DFlash

来源：[vllm-project/vllm#45181](https://github.com/vllm-project/vllm/pull/45181)

状态：merged，2026-06-21 合入。

可视化图解：[DFlash target/draft KV page size 不一致时怎么对齐](../../visuals/vllm-45181-dflash-kv-page-padding.html)

## 一句话

这个 PR 让 vLLM 的 KV cache 基础设施支持 DFlash speculative decoding 中 target model 和 draft model 使用不同 KV page size：能整除时继续放大 block size，不能整除时给较小 page 做物理 padding，并用 `torch.as_strided` 让 attention kernel 跳过 padding 区。

## 先把问题说成人话

KV cache 可以想成一排仓库格子。每个格子保存一段 token 的 K/V。

```text
page = 一个 KV cache block 的物理存储单位
block_size = 这个 page 里逻辑上放多少 token
page_size_bytes = 这个 page 在显存里占多少字节
```

DFlash speculative decoding 里可能有：

```text
target model: KV head size = 192
draft model:  KV head size = 128
```

同样 block_size 下，target 的 KV page 比 draft 大。vLLM 想把不同层/模型的 KV cache 放进统一管理的 cache group，就要让 page size 对齐。

## 原来的统一方法

旧逻辑只有一种办法：

```text
如果小 page 能整除大 page：
  增大小模型的 block_size
  让它一页也占到大 page 的字节数
```

举例：

```text
大 page = 1024 bytes
小 page = 512 bytes
ratio = 2
小模型 block_size *= 2
```

这样逻辑上小模型一页放更多 token，物理 page size 就一样了。

问题是 DFlash 的 192 vs 128 是 3:2，不是整数倍关系：

```text
大 page / 小 page = 192 / 128 = 1.5
```

你不能把 block_size 乘以 1.5。旧逻辑就会报 `NotImplementedError`。

## 新方法：不能整除就 padding

新逻辑在：

```text
vllm/v1/core/kv_cache_utils.py::unify_kv_cache_spec_page_size
```

简化后是：

```python
if max_page_size % layer_page_size == 0:
    ratio = max_page_size // layer_page_size
    new_spec = replace(layer_spec, block_size=layer_spec.block_size * ratio)
elif isinstance(layer_spec, AttentionSpec) and layer_spec.indexes_kv_by_block_stride:
    new_spec = replace(layer_spec, page_size_padded=max_page_size)
else:
    raise NotImplementedError
```

小白记法：

```text
能用 block_size 对齐：优先不浪费空间
不能整除：保留逻辑 block_size，只把物理 page 补到一样大
```

## padding 后 kernel 怎么不读到空洞？

这是最关键的代码点：

```text
vllm/v1/worker/gpu/attn_utils.py::_reshape_attention_kv_cache
```

当 `page_size_padded is not None` 时，不能简单 `.view()`，因为 `.view()` 会把 padding 也当作连续有效数据。

PR 改成 `torch.as_strided`：

```python
page_stride = kv_cache_spec.page_size_bytes // dtype_size
strides = list(torch.empty(permuted_kv_cache_shape).stride())
strides[num_blocks_dim] = page_stride

kv_cache = torch.as_strided(
    kv_raw_tensor.view(dtype),
    size=permuted_kv_cache_shape,
    stride=tuple(strides),
)
```

这段代码的意思是：

```text
每次从 block 0 跳到 block 1
不要跳真实有效 page 的长度
而要跳 padded page 的长度
```

这样 block 和 block 之间的 padding 区自然被跨过去。

## 为什么要 `indexes_kv_by_block_stride`

不是所有 attention backend 都能接受这种“block 之间有空洞”的布局。

PR 增加了：

```text
AttentionSpec.indexes_kv_by_block_stride
AttentionBackend.indexes_kv_by_block_stride()
```

它表示 backend 是否按 block stride 访问 KV cache。只有 backend 明确支持这种布局时，vLLM 才允许 padding。

小白记法：

```text
padding 不是白送的
kernel 必须知道每个 block 的起点按 stride 跳
```

## 为什么不是直接复制成连续 tensor？

因为复制会带来额外显存写和读，还会破坏 KV cache 管理的统一性。

`as_strided` 的好处是：

```text
不复制数据
不重新分配大 tensor
只创建一个“怎么看这块内存”的 view
```

它的风险是 stride 一旦算错，kernel 会读错位置。所以 PR 加了很多 stride/offset 测试。

## 测试覆盖了什么

PR 主要加了这些测试：

1. target head size 192、draft head size 128，3:2 非整数比例时使用 padding。
2. backend 不支持 `indexes_kv_by_block_stride` 时仍然报错。
3. FlashAttention 风格 layout 的 padded page stride。
4. HND stride order。
5. DiffKV 风格 layout。
6. quantized KV cache 的 per-token/head scale 存储。

这些测试说明 PR 的重点不是“只让一个例子跑通”，而是让 KV cache reshape 的 stride 逻辑在多种 backend layout 下成立。

## 为什么会优化

更准确地说，这个 PR 首先是“解锁能力”，然后才谈性能。

它让 DFlash 这种 target/draft KV page size 不同的组合能复用 vLLM 的统一 KV cache 管理，而不是因为 page size 3:2 就退回不能支持。

性能上的好处来自：

```text
不用复制/重排 KV cache
不用为非整数 page ratio 走特殊慢路径
保留统一 cache group 的内存管理
```

但 PR 没有给出明确端到端 speedup 数字。收益主要是支持 DFlash/MiMo 类场景和避免额外 materialization。

## 代码阅读路线

建议按这个顺序读：

1. `vllm/v1/core/kv_cache_utils.py::unify_kv_cache_spec_page_size`
2. `vllm/v1/kv_cache_interface.py::AttentionSpec`
3. `vllm/v1/attention/backend.py::indexes_kv_by_block_stride`
4. `vllm/v1/worker/gpu/attn_utils.py::_reshape_attention_kv_cache`
5. `tests/v1/core/test_kv_cache_utils.py`
6. `tests/v1/worker/test_attn_utils.py`

## 小白快速记忆

背这句话：

```text
vLLM #45181 = DFlash 里 target/draft KV page 对不齐时，小 page 不改逻辑 token 数，只在物理内存后面补空洞。
```

再记三件事：

```text
3:2 不能靠 block_size 整除
page_size_padded 让物理 page 一样大
as_strided 让 kernel 跳过 padding
```

## 可能问题

### 1. padding 会不会浪费显存？

会浪费一部分物理 page 空间。比如 draft page 128 对齐到 target page 192，就有 1/3 左右的空洞。但它换来的是统一 KV cache layout 和不复制数据。

### 2. 为什么不直接把 draft block_size 改成 24？

如果原 block_size 是 16，乘以 1.5 才能从 128 对齐到 192。但 block_size 必须是整数，而且 attention backend、scheduler、KV 管理通常都假设 block_size 是明确的 token 数。直接改成非整数不存在；改成别的整数又不等价。

### 3. `page_size_padded` 和 `real_page_size_bytes` 有什么区别？

```text
real_page_size_bytes:
  真实有效 KV 数据占多少字节

page_size_padded / page_size_bytes:
  对齐后物理上每页占多少字节
```

kernel 读有效数据，但跳到下一页时按 padded page 的 stride 跳。

### 4. 为什么必须是 num-blocks-first layout？

因为 padding 是加在 block/page 之间的。如果 `num_blocks` 不是外层物理维度，简单改 block stride 可能不能正确跳过 padding，还可能影响 K/V/head/token 的内部连续性。

### 5. 其他模型想用这个能力，要检查什么？

至少检查：

1. target/draft KV head size 和 page size 比例。
2. attention backend 是否支持 `indexes_kv_by_block_stride`。
3. KV cache stride order 是否明确。
4. quantized KV scale 是否也在正确 stride 下。
5. page padding 的显存开销是否可接受。

### 6. 这个 PR 和 MoE 有关系吗？

不是 MoE kernel 优化。它属于 speculative decoding + KV cache layout 基础设施。它和前面 MoE PR 的共同点是：都在处理“高性能 kernel 对 shape/layout 很挑剔”的问题。
