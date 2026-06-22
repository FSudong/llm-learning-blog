# vLLM #45424: pin memory before async H2D copy

来源：[vllm-project/vllm#45424](https://github.com/vllm-project/vllm/pull/45424)

状态：merged，2026-06-21 合入。

可视化图解：[为什么 non_blocking H2D 之前必须 pin memory](../../visuals/vllm-45424-pinned-memory-async-h2d.html)

## 一句话

这个 PR 清理 vLLM 里大量 CPU -> GPU tensor copy：在 `non_blocking=True` 的 H2D copy 前确保 CPU tensor 是 pinned memory，并把常见路径收敛到 `async_tensor_h2d` / `PIN_MEMORY` 工具，减少潜在的 CPU/GPU stream sync。

## 先把 pinned memory 说成人话

CPU 普通内存可以被操作系统移动或换页。GPU 做 DMA 传输时，不喜欢这种“可能被移动”的内存。

pinned memory 也叫 page-locked memory：

```text
操作系统承诺这块 CPU 内存不会被换页/移动
GPU 可以更直接地从这里异步搬数据
```

所以 PyTorch 里的 `non_blocking=True` 有个重要前提：

```text
CPU source tensor 最好是 pinned memory
```

否则你写了 `non_blocking=True`，底层也可能需要临时 staging 或同步，异步效果打折。

## 原来的问题

vLLM 里有很多类似代码：

```python
torch.from_numpy(cu_seqlens).to(device, non_blocking=True)
```

或者：

```python
gpu_buffer.copy_(torch.from_numpy(req_id_per_token), non_blocking=True)
```

看起来是异步 copy，但源 tensor 来自普通 CPU memory，不一定 pinned。

这类代码的问题是：

```text
你要求异步 H2D
但源内存不满足高效异步传输条件
底层可能插入同步或额外 copy
```

在 decode / attention metadata / speculative decoding 这种高频路径里，很多小同步会放大。

## 新路径

PR 做了两类改动。

### 1. 显式 pin 再 copy

例如 sparse MLA 路径里：

```python
self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
    torch.from_numpy(req_id_per_token).pin_memory(),
    non_blocking=True,
)
```

含义是：

```text
numpy -> CPU tensor
CPU tensor pin_memory()
再 non_blocking copy 到 GPU buffer
```

### 2. 用统一工具 `async_tensor_h2d`

很多地方从：

```python
torch.tensor(indices, device=self.device)
```

或：

```python
torch.from_numpy(x).to(device, non_blocking=True)
```

改成：

```python
async_tensor_h2d(indices, device=self.device)
```

这个工具函数的价值是把“创建 CPU tensor、pin、H2D、dtype/device 处理”集中起来，避免每个调用点自己写一份容易漏 pin 的代码。

## 一个很关键的小细节：避免 GPU -> CPU 反向同步

PR 还改了 pooling 里的 `repeat_interleave`。

旧逻辑为了避免 GPU `repeat_interleave` 推断输出长度时同步，先在 CPU 上构造 `segment_ids`，再上传：

```python
segment_ids = torch.repeat_interleave(
    torch.arange(num_seqs, dtype=torch.long),
    prompt_lens_cpu,
).to(hidden_states.device, non_blocking=True)
```

新逻辑先把 `prompt_lens` pinned async 上传到 GPU，然后在 GPU 上做 repeat，并显式给 `output_size`：

```python
prompt_lens = async_tensor_h2d(
    prompt_lens_cpu,
    device=hidden_states.device,
    dtype=torch.int64,
)

segment_ids = torch.repeat_interleave(
    torch.arange(num_seqs, device=hidden_states.device, dtype=torch.long),
    prompt_lens,
    output_size=int(prompt_lens_cpu.sum()),
)
```

重点是：

```text
output_size 用 CPU 已知的 prompt_lens_cpu.sum()
不要让 GPU 算完再告诉 CPU 输出长度
```

这和 SGLang #28782 里避免 `.sum().item()` 是同一类思想：不要在热路径里让 CPU 等 GPU 回答一个小数字。

## `PIN_MEMORY` 的作用

PR 还把很多 `is_pin_memory_available()` 的调用标准化，统一使用：

```text
vllm.utils.torch_utils.PIN_MEMORY
```

这样构造 CPU buffer 时可以统一判断当前平台是否适合 pin memory，而不是到处散落不同判断。

## 为什么会优化

收益来自减少同步和减少隐式 staging：

```text
普通 CPU memory + non_blocking=True:
  可能并不是真异步
  可能触发额外 copy 或等待

pinned CPU memory + non_blocking=True:
  更容易让 H2D copy 和 GPU compute overlap
```

这个 PR 没有给出明确 speedup 数字。它的目标更偏基础设施卫生：

```text
让所有声称 async H2D 的路径真的具备 async 条件
```

## 影响范围

PR 改了很多调用点，包括：

```text
attention metadata
FlexAttention doc ids
MLA sparse metadata
spec decode token indices
ngram proposer GPU tensors
structured output bitmask/index tensor
pooling prompt_lens / segment_ids
multimodal input reduce
gpu_model_runner
tests around input batch pin_memory
```

这说明它不是某个模型特化，而是 vLLM 热路径里 CPU/GPU copy 规范的清理。

## 代码阅读路线

建议按这个顺序读：

1. `vllm/utils/torch_utils.py` 中的 `PIN_MEMORY` / `async_tensor_h2d`
2. `vllm/v1/worker/gpu_model_runner.py`
3. `vllm/v1/attention/backends/utils.py`
4. `vllm/models/deepseek_v4/sparse_mla.py`
5. `vllm/model_executor/layers/pooler/seqwise/methods.py`
6. `tests/v1/worker/test_gpu_input_batch.py` 相关构造参数删除

## 小白快速记忆

背这句话：

```text
vLLM #45424 = 不要只写 non_blocking=True；CPU 源 tensor 要先 pin，H2D 才更可能真异步。
```

再记一个图：

```text
CPU 普通内存
  -> 可能 staging / sync
  -> GPU

CPU pinned memory
  -> async H2D copy
  -> GPU
```

## 可能问题

### 1. `non_blocking=True` 不是已经异步了吗？

不是保证。它只是告诉 PyTorch“如果条件允许，请异步”。CPU 到 GPU copy 要真正异步，CPU 源内存通常需要 pinned。否则底层可能仍要同步或先复制到 pinned staging buffer。

### 2. 为什么不把所有 CPU tensor 都 pin？

pinned memory 不是免费的。它会占用 page-locked 内存，太多会影响系统内存管理。所以应该 pin 热路径上确实要 H2D 的 tensor，而不是无脑 pin 所有数据。

### 3. `.pin_memory()` 本身会不会有成本？

有成本。如果每次都临时 pin 很大 tensor，也可能不划算。这个 PR 的重点是把 vLLM 已经在热路径里要异步上传的小 metadata/index tensor 做正确，长期更应该复用 pinned buffer。

### 4. 这和 CUDA Graph 有什么关系？

CUDA Graph 需要减少 replay 前后的动态同步。H2D 小 metadata 如果隐式同步，会削弱 graph/异步执行收益。这个 PR 不直接“capture graph”，但让 graph 周边的数据搬运更干净。

### 5. 为什么 pooling 里要给 `output_size`？

如果 `repeat_interleave` 需要根据 GPU 上的 `prompt_lens` 推断输出长度，CPU 可能要等 GPU 结果。`output_size=int(prompt_lens_cpu.sum())` 用 CPU 已知数据直接告诉它输出多长，避免这个同步点。

### 6. 其他项目怎么迁移这个思想？

检查所有热路径：

```text
.to("cuda", non_blocking=True)
.copy_(cpu_tensor, non_blocking=True)
torch.from_numpy(...).to(device)
```

然后问三个问题：

1. CPU source 是 pinned 吗？
2. 这个 copy 是否在 decode/step 级热路径？
3. 有没有 `.item()` / GPU shape 推断让 CPU 等 GPU？

如果答案暴露问题，就应该改成 pinned buffer、统一 H2D helper 或 CPU 已知 output size。
