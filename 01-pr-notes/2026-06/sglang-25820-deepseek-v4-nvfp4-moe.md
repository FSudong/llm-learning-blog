# SGLang #25820: DeepSeek-V4 NVFP4 MoE support

来源：[sgl-project/sglang#25820](https://github.com/sgl-project/sglang/pull/25820)

状态：merged，2026-06-22 合入。

可视化图解：[DeepSeek-V4 的 NVFP4 MoE 是怎么被框架接上的](../../visuals/sglang-25820-deepseek-v4-nvfp4-moe.html)

## 一句话

这个 PR 不是只写了一个新 kernel，而是把 `nvidia/DeepSeek-V4-Pro-NVFP4` 这种“普通层 FP8、MoE expert NVFP4”的混合量化模型完整接进 SGLang：自动识别 checkpoint，给 FusedMoE 选择 NVFP4 方法，默认切到 `flashinfer_trtllm_routed` MoE runner，并补齐 routing scale、SwiGLU clamp 和 docs benchmark。

## 先用小白图记住

```text
模型配置
  -> 发现 moe_quant_algo = NVFP4
  -> 生成 HybridFp8NvFp4Config
  -> 普通 Linear/Attention 继续走 FP8
  -> FusedMoE expert 走 NVFP4
  -> FlashInfer TRTLLM routed MoE kernel 计算
```

最容易混淆的一点：它不是把整个 DeepSeek-V4 都变成 NVFP4。更准确地说：

```text
非 MoE 层：FP8 路径
MoE expert 权重：NVFP4 路径
```

## 原来缺什么

DeepSeek-V4 NVFP4 checkpoint 的量化信息不是一个简单字段就能表达完。`config.json` 会告诉你这是 mixed precision，并且 MoE 使用 NVFP4；更细的 `group_size`、`exclude_modules` 等信息在 `hf_quant_config.json` 里。

如果框架只把它当成普通 FP8 checkpoint，会出现三个问题：

1. FusedMoE 不会选择 NVFP4 expert 方法。
2. MoE runner 不一定切到支持 FP4 MoE 的 backend。
3. DeepSeek-V4 的 routing scale、SwiGLU clamp 等细节可能没有传到 kernel。

所以这个 PR 的目标是“把模型跑通且跑在正确快路径上”，不是单点微调。

## 代码变动主线

### 1. `model_config.py`：先识别这是 NVFP4 MoE

关键字段是：

```python
self.nvfp4_moe_meta = None

if (
    hybrid_quant_cfg is not None
    and hybrid_quant_cfg.get("quant_algo") == "MIXED_PRECISION"
    and str(hybrid_quant_cfg.get("moe_quant_algo", "")).upper() == "NVFP4"
):
    self.nvfp4_moe_meta = {
        "group_size": int(...),
        "exclude_modules": list(...),
    }
```

可以把它理解成：模型启动时先贴一张标签。

```text
这个 checkpoint:
  普通层按 FP8 看
  MoE expert 按 NVFP4 看
  group_size = ...
  哪些 module 不走 NVFP4 = ...
```

### 2. `loader.py`：把普通 FP8 配置包成 hybrid 配置

旧世界只有一个 `Fp8Config`。新世界需要告诉 layer dispatch：

```text
看到普通 Linear/Attention:
  继续使用 Fp8Config

看到 FusedMoE:
  改用 ModelOptNvFp4FusedMoEMethod
```

因此 loader 里会构造：

```python
ModelOptFp4Config(
    is_checkpoint_nvfp4_serialized=True,
    group_size=nvfp4_meta["group_size"],
    exclude_modules=...
)

HybridFp8NvFp4Config(fp8_config, nvfp4_config)
```

`HybridFp8NvFp4Config` 的作用就是“分流器”：普通层交给继承来的 FP8 逻辑，FusedMoE 交给 NVFP4 逻辑。

### 3. `modelopt_quant.py`：FusedMoE 走 NVFP4 方法

核心类：

```text
HybridFp8NvFp4Config(Fp8Config)
```

核心行为：

```text
if layer is FusedMoE and not excluded:
    return ModelOptNvFp4FusedMoEMethod(...)
else:
    return super().get_quant_method(...)
```

这就是“混合量化”真正落地的地方。

### 4. `deepseek_v4_hook.py`：默认选择 FP4 MoE backend

PR 里增加了自动选择：

```python
if server_args.moe_runner_backend == "auto"
   and model_config.nvfp4_moe_meta is not None:
    server_args.moe_runner_backend = "flashinfer_trtllm_routed"
```

小白可以这样记：

```text
发现 DeepSeek-V4 NVFP4 MoE
  -> 别走通用 MoE runner
  -> 默认用 flashinfer_trtllm_routed
```

因为这个 backend 才是这条 NVFP4 routed MoE 快路径。

### 5. `hash_topk.py`：补 routing scale 语义

DeepSeek-V4 routing 里有一个细节：有些模型要求把 `routed_scaling_factor` 作用到输出上。

PR 把它接到了 top-k weights：

```python
if self.apply_routed_scaling_factor_on_output:
    topk_weights = topk_weights * self.routed_scaling_factor
```

但是它和 fused shared experts 组合还不支持，所以 PR 明确报错：

```text
HashTopK + apply_routed_scaling_factor_on_output
不能和 fused shared experts 同时用
```

这类报错很重要：它避免“能跑但语义悄悄错”。

### 6. `flashinfer_trtllm.py`：传递 SwiGLU clamp

DeepSeek-V4 的 expert MLP 使用 SwiGLU。PR 补了 `gemm1_clamp_limit`，让 kernel 在正确位置做 clamp。

简化理解：

```text
GEMM1 输出
  -> 按 DeepSeek-V4 要求做 clamp
  -> SwiGLU activation
  -> GEMM2
```

如果少了这个参数，kernel 数学语义可能和模型定义不一致。

## 为什么会有优化

收益不是来自“少一个 Python if”，而是来自这三件事：

1. NVFP4 expert 权重更小，MoE GEMM 读权重的带宽压力更低。
2. `flashinfer_trtllm_routed` 直接处理 routed MoE，避免落回更通用的慢路径。
3. 混合量化只让 MoE expert 用 NVFP4，普通层仍复用成熟 FP8 路径，减少实现范围和风险。

可以把它看成：

```text
不是全模型重写
而是把最重的 MoE expert 部分接到 FP4 专用快车道
```

## PR 给出的效果

PR 文档里给了 DeepSeek-V4 NVFP4 相关 benchmark。摘几个最容易记的数：

| 硬件/配置 | workload | TTFT | TPOT | tokens/s/GPU | accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| GB200 flash single | random 8192/1024, concurrency 1 | 323.85 ms | 3.62 ms | 496 | GSM8K 96.66% |
| GB200 flash single | random 8192/1024, concurrency 16 | 397.31 ms | 8.11 ms | 3663 | GSM8K 96.66% |
| GB300 flash single | random 8192/1024, concurrency 1 | 361.72 ms | 3.62 ms | 480 | GSM8K 96.44% |
| GB300 flash single | random 8192/1024, concurrency 16 | 422.96 ms | 8.19 ms | 3733 | GSM8K 96.44% |

PR 描述还给了一个 accuracy/throughput 测试：

```text
model: nvidia/DeepSeek-V4-Pro-NVFP4
Accuracy: 0.965
Invalid: 0.000
Latency: 53.335 s
Output throughput: 2149.225 token/s
```

注意：这些是 PR 给出的运行结果，不是和某个明确 baseline 的 A/B speedup。这个 PR 的重点是“支持并走快路径”，不是单独证明某个 micro-kernel 加速了多少倍。

## 代码阅读路线

建议按这个顺序读：

1. `python/sglang/srt/configs/model_config.py`
2. `python/sglang/srt/model_loader/loader.py`
3. `python/sglang/srt/layers/quantization/modelopt_quant.py`
4. `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
5. `python/sglang/srt/layers/moe/hash_topk.py`
6. `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
7. `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`

## 小白快速记忆

背这张三段式：

```text
识别：model_config 读出 NVFP4 MoE meta
分流：HybridFp8NvFp4Config 让 FusedMoE 走 NVFP4
执行：flashinfer_trtllm_routed 负责 FP4 routed MoE kernel
```

再记一句话：

```text
SGLang #25820 = DeepSeek-V4 混合量化 checkpoint 的 MoE FP4 快路径接线。
```

## 可能问题

### 1. 这是不是和 vLLM #44667 一样？

不一样。vLLM #44667 主要讲 kernel 内 inline dequant，重点是“边解边 GEMM”。SGLang #25820 更像端到端接入：识别 checkpoint、创建 hybrid quant config、选择 MoE backend、补模型语义参数。

### 2. 为什么不全模型都 NVFP4？

因为 checkpoint 本身是混合量化。普通层和 MoE expert 的量化格式、scale、kernel 支持都不同。强行统一反而会增加错误和 fallback。

### 3. `exclude_modules` 是干什么的？

不是所有模块都适合或已经被 NVFP4 量化。`exclude_modules` 用来告诉框架：这些模块不要走 NVFP4 方法，继续走别的路径。

### 4. 为什么要默认切 `flashinfer_trtllm_routed`？

因为 NVFP4 MoE 需要 backend 能理解 FP4 expert 权重和 routed MoE 输入。通用 backend 可能能跑但不是这条快路径，也可能不支持对应权重布局。

### 5. 其他模型也想上 NVFP4 MoE，要做什么？

至少要补：

1. checkpoint 配置解析：能拿到 quant algo、group size、scale layout、exclude list。
2. 权重 loader：能把 Hugging Face 权重名映射到框架内部 FusedMoE 权重。
3. quant method：FusedMoE 能选择正确的 FP4 method。
4. backend kernel：支持该模型的 top-k、activation、scale、TP/EP shard。
5. correctness test：reference 路径和新路径对齐。
6. benchmark：tokens、batch、TP/EP、accuracy 都要覆盖。

### 6. 这里的优化风险在哪里？

风险主要是语义错而不是编译错：

```text
routing scale 少乘一次
SwiGLU clamp 位置错
某些 module 不该用 NVFP4 却用了
weight scale group size 解析错
```

所以读这类 PR，一定要同时看 config 解析、loader、quant method、kernel 参数和 accuracy。
