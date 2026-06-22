# 2026-06 PR 精读

## 已完成

- [SGLang #28450: Fuse shared-expert append + DeepEP remap](sglang-28450-fused-append-remap-deepep.md)
  - [可视化图解：top-k 表如何一步步变化](../../visuals/sglang-28450-moe-routing-visual.html)
- [vLLM #44667: Fuse NVFP4 dequantization with MoE compute](vllm-44667-nvfp4-moe-fused-dequant.md)
  - [可视化图解：NVFP4 权重如何边解边乘](../../visuals/vllm-44667-nvfp4-moe-fused-dequant.html)
- [SGLang #25820: Support NVFP4 MoE for DeepSeek-V4](sglang-25820-deepseek-v4-nvfp4-moe.md)
  - [可视化图解：DeepSeek-V4 NVFP4 MoE 接线链路](../../visuals/sglang-25820-deepseek-v4-nvfp4-moe.html)
- [SGLang #28782: FlashInfer CUDA Graph for EAGLE draft-extend](sglang-28782-eagle-draft-extend-flashinfer-cudagraph.md)
  - [可视化图解：固定 draft-extend qo layout](../../visuals/sglang-28782-eagle-draft-extend-cudagraph.html)
- [vLLM #45181: Mixed KV page sizes for DFlash](vllm-45181-dflash-mixed-kv-page-sizes.md)
  - [可视化图解：KV page padding + as_strided](../../visuals/vllm-45181-dflash-kv-page-padding.html)
- [vLLM #45232: FlexAttention custom mask full CUDAGraph](vllm-45232-flexattention-custom-mask-cudagraph.md)
  - [可视化图解：custom mask metadata 前移](../../visuals/vllm-45232-flexattention-mask-cudagraph.html)
- [vLLM #45424: Pinned memory before async H2D](vllm-45424-pinned-memory-async-h2d.md)
  - [可视化图解：non_blocking H2D 为什么要 pin memory](../../visuals/vllm-45424-pinned-memory-async-h2d.html)

## 当前主题

MoE 推理加速和推理 runtime 基础设施：

- 减少 top-k 后处理阶段的小 kernel launch
- 避免量化 MoE 权重全量反量化
- 支持 mixed FP8+NVFP4 checkpoint 的 MoE 快路径接入
- 让 speculative decoding / attention mask metadata 更适合 CUDA Graph
- 处理 KV cache page layout 和 CPU->GPU metadata copy 的隐藏成本
- 明确优化收益来自哪里
- 总结迁移到其他 MoE 模型时需要补的条件
