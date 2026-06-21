# 2026-06 PR 精读

## 已完成

- [SGLang #28450: Fuse shared-expert append + DeepEP remap](sglang-28450-fused-append-remap-deepep.md)
- [vLLM #44667: Fuse NVFP4 dequantization with MoE compute](vllm-44667-nvfp4-moe-fused-dequant.md)

## 当前主题

MoE 推理加速：

- 减少 top-k 后处理阶段的小 kernel launch
- 避免量化 MoE 权重全量反量化
- 明确优化收益来自哪里
- 总结迁移到其他 MoE 模型时需要补的条件

