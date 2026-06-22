# SGLang #28782: FlashInfer CUDA Graph for EAGLE draft-extend

来源：[sgl-project/sglang#28782](https://github.com/sgl-project/sglang/pull/28782)

状态：merged，2026-06-21 合入。

可视化图解：[EAGLE draft-extend 怎么被 FlashInfer CUDA Graph 捕获](../../visuals/sglang-28782-eagle-draft-extend-cudagraph.html)

## 一句话

这个 PR 让 EAGLE speculative decoding 的 `draft-extend` 阶段在 FlashInfer attention backend 下也能走 CUDA Graph：把动态的 query layout 固定成 padded tree width，提前准备 FlashInfer metadata，并移除一次 `.sum().item()` 造成的 device-to-host 同步。

## 先把 EAGLE 说成人话

EAGLE 是 speculative decoding 的一种。可以粗略理解成：

```text
draft model 先猜几个未来 token
target model 再验证这些 token
接受的 token 越多，生成越快
```

其中 `draft-extend` 是 draft 侧继续扩展候选树的一步。它每轮都会跑 attention。旧问题是：FlashInfer backend 在这个阶段没法完整 graph capture，于是每轮还是有一串 eager kernel launch 和 metadata 构造成本。

## 旧路径的问题

CUDA Graph 喜欢固定形状、固定内存地址、固定执行拓扑。

但 EAGLE draft-extend 里有两个动态点：

```text
num_accept_tokens:
  每个请求实际接受几个 token，是运行时结果

prefix_lens.sum().item():
  从 GPU tensor 读回 CPU，形成 D2H 同步
```

这两个点会破坏 graph replay 的稳定性。

可以想成：

```text
graph capture 想拍一张固定动作录像
但旧代码每轮都说：等我先看一下 GPU 上这轮到底有几个 token
```

## 新路径的核心

PR 做了三件事。

### 1. FlashInfer backend 加入 draft-extend graph 白名单

入口在：

```text
python/sglang/srt/speculative/eagle_worker_v2.py
```

新逻辑把 `FlashInferAttnBackend` 纳入 `supports_cuda_draft_extend_graph`，但有一个限制：

```python
flashinfer_graph_supported = (
    isinstance(draft_extend_backend, FlashInferAttnBackend)
    and self.server_args.speculative_token_map is None
)
```

如果启用了 FR-Spec 的 `speculative_token_map`，FlashInfer draft-extend graph 先 fallback 到 eager。

### 2. 固定 qo layout

关键改动在：

```text
python/sglang/srt/speculative/eagle_info.py
```

旧逻辑：

```python
qo_indptr[1:] = torch.cumsum(self.num_accept_tokens, dim=0)
```

这表示每个请求有多少 query row 取决于 `num_accept_tokens`。这是动态的。

新逻辑：

```python
qo_indptr = torch.arange(
    0,
    (bs + 1) * self.num_tokens_per_req,
    step=self.num_tokens_per_req,
    dtype=torch.int32,
    device=device,
)
```

意思是每个请求都按固定 `num_tokens_per_req` 行来摆放。实际没用满的 token 行用 padding 占位。

小白记法：

```text
旧：每个请求真实接受几个，就摆几行
新：每个请求都摆固定宽度，不够就空着
```

固定宽度之后，CUDA Graph 就能捕获和重放。

### 3. 去掉一次 D2H 同步

关键改动在：

```text
python/sglang/srt/layers/attention/flashinfer_backend.py
```

旧逻辑：

```python
paged_kernel_lens_sum = paged_kernel_lens.sum().item()
```

`.item()` 会把 GPU tensor 的结果读回 CPU。CPU 必须等 GPU 算完这个 sum，所以它是一个同步点。

新逻辑：

```python
if extend_prefix_lens_cpu is not None:
    paged_kernel_lens_sum = sum(extend_prefix_lens_cpu)
else:
    paged_kernel_lens_sum = paged_kernel_lens.sum().item()
```

如果 CPU 本来就知道 prefix lens，就直接在 CPU 上求和，不再从 GPU 读回。

## FlashInfer metadata 是怎么接上的

PR 在 `flashinfer_backend.py` 增加了 draft-extend 分支。

capture 阶段：

```python
elif forward_mode.is_draft_extend_v2():
    prefill_wrappers = self._create_prefill_wrappers(
        bs,
        use_custom_mask=False,
    )
    self.draft_extend_cuda_graph_metadata[bs] = prefill_wrappers
    self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
```

replay / out-of-graph metadata 初始化阶段：

```python
elif forward_mode.is_draft_extend_v2():
    self.indices_updater_prefill.update(
        req_pool_indices[:bs],
        seq_lens[:bs],
        seq_lens_cpu[:bs],
        seq_lens_sum,
        prefix_lens=None,
        prefill_wrappers=self.draft_extend_cuda_graph_metadata[bs],
        use_ragged=False,
        spec_info=spec_info,
    )
```

可以理解成：

```text
capture 时先为 batch size = bs 拍好一套 FlashInfer prefill wrapper
replay 时直接复用这套 wrapper，只更新必要 metadata
```

## 为什么会优化

收益来自三层：

1. draft-extend attention 不再每轮 eager launch 一堆小 kernel。
2. 固定 qo layout 后，CUDA Graph 可以复用同一张执行图。
3. 移除 `sum().item()`，减少 CPU 等 GPU 的同步点。

它优化的是 speculative decoding 的调度/attention 执行成本，不改变 EAGLE 接受 token 的算法。

## PR 给出的效果

PR 描述给出的线上实验：

```text
request-rate: 8
model: Llama-2-7B + EAGLE
speculative-eagle-topk: 4
online median E2E latency: 约 10% 改善
accept length: 基本不变
```

所以这不是“多接受 token”带来的收益，而是“同样的 speculative 结果，运行开销更低”。

## 代码阅读路线

建议按这个顺序读：

1. `python/sglang/srt/speculative/eagle_worker_v2.py`
2. `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`
3. `python/sglang/srt/speculative/eagle_info.py`
4. `python/sglang/srt/layers/attention/flashinfer_backend.py`

重点盯住三个变量：

```text
num_tokens_per_req
qo_indptr
extend_prefix_lens_cpu
```

## 小白快速记忆

背这句话：

```text
SGLang #28782 = 把 draft-extend 从“每轮动态摆 token”改成“固定 padded 宽度”，让 FlashInfer attention 能被 CUDA Graph 重放。
```

再背三个关键词：

```text
fixed qo layout
FlashInfer prefill wrapper
avoid sum().item()
```

## 可能问题

### 1. 为什么固定 `num_tokens_per_req` 不会算错？

因为它只是固定内存和 query row 的布局，不等于所有位置都产生有效 token。无效位置可以通过 padding 和后续逻辑忽略。CUDA Graph 需要的是固定形状，语义上仍只接受真实有效 token。

### 2. `qo_indptr` 是什么？

可以把它当成“每个请求的 query row 从哪里开始、到哪里结束”的边界表。

```text
bs = 3, num_tokens_per_req = 4
qo_indptr = [0, 4, 8, 12]
```

这表示：

```text
req0 用 query row 0..3
req1 用 query row 4..7
req2 用 query row 8..11
```

旧版每个请求边界取决于真实接受 token 数；新版边界固定。

### 3. 为什么 `.sum().item()` 会拖慢？

`sum()` 在 GPU 上算，`.item()` 要把结果拿回 CPU。CPU 想拿这个值就必须等 GPU 完成，因此形成同步。decode 循环里每步都来一次，会很伤。

### 4. 为什么 FR-Spec / `speculative_token_map` 不能一起 graph？

PR 里说明 FlashInfer draft-extend graph buffers 是按完整 draft vocab 尺寸设计的；FR-Spec 使用 reduced hot vocab，buffer 语义不匹配，所以先 fallback 到 eager。

### 5. 其他 backend 或模型想上这个优化，要检查什么？

至少检查：

1. draft-extend 的 query layout 能不能固定。
2. metadata builder 能不能在 capture 前准备好。
3. replay 时所有 tensor shape 是否稳定。
4. 是否还有 `.item()`、`.cpu()`、动态 allocation。
5. reduced vocab、custom mask、ragged prefill 是否会改变 buffer shape。

### 6. 这个 PR 和 SGLang #28450 的共同点是什么？

共同点都是“让执行路径更适合 graph/fusion”：

```text
#28450: top-k 表一次写最终布局，减少小 kernel
#28782: draft-extend 固定布局，进入 CUDA Graph
```

一个发生在 MoE top-k 后处理，一个发生在 speculative attention。
