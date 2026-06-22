# LLM Learning Blog

这个仓库用于长期记录 LLM 系统、推理框架、MoE 加速、训练/推理并行和高性能 kernel 的源码学习笔记。

当前第一组主题是：SGLang / vLLM 最近 MoE 推理加速 PR 精读。

## 阅读入口

- [MoE 推理加速知识拓扑](00-knowledge-map/moe-inference-acceleration-topology.md)
- [2026-06 PR 精读索引](01-pr-notes/2026-06/README.md)
- [SGLang #28450 可视化图解](visuals/sglang-28450-moe-routing-visual.html)
- [vLLM #44667 可视化图解](visuals/vllm-44667-nvfp4-moe-fused-dequant.html)
- [SGLang #25820 可视化图解](visuals/sglang-25820-deepseek-v4-nvfp4-moe.html)
- [SGLang #28782 可视化图解](visuals/sglang-28782-eagle-draft-extend-cudagraph.html)
- [vLLM #45181 可视化图解](visuals/vllm-45181-dflash-kv-page-padding.html)
- [vLLM #45232 可视化图解](visuals/vllm-45232-flexattention-mask-cudagraph.html)
- [vLLM #45424 可视化图解](visuals/vllm-45424-pinned-memory-async-h2d.html)
- [MoE 优化模式库](02-patterns/moe-optimization-patterns.md)
- [PR 源码阅读模板](03-code-reading/pr-reading-template.md)
- [把 MoE 优化迁移到其他模型的 checklist](04-model-porting/moe-porting-checklist.md)
- [GitHub PR 来源索引](sources/github-pr-index.md)

## 目录约定

```text
00-knowledge-map/      知识拓扑和总览，用来建立系统地图
01-pr-notes/           具体 PR 精读，按年月归档
02-patterns/           从多个 PR 抽象出的通用优化模式
03-code-reading/       源码阅读方法、模板、复盘格式
04-model-porting/      把优化迁移到新模型/新硬件时的 checklist
sources/               原始链接、版本信息、外部资料索引
```

## 写作原则

每篇技术笔记尽量回答六个问题：

1. 这个 PR 解决的是 MoE pipeline 的哪一段？
2. 原始瓶颈是 kernel launch、HBM、通信、负载不均，还是算力不足？
3. 代码改动的入口函数在哪里？
4. 新路径和旧路径在执行链路上有什么差异？
5. 收益来自哪里，PR 给出的效果是多少？
6. 如果换成其他模型、其他硬件、其他量化格式，要补哪些条件？

## 当前学习主线

第一阶段先把 MoE 推理拆成四块：

- 路由和 top-k 后处理
- dispatch / all-to-all / expert parallel
- expert GEMM / fused MoE kernel
- combine / reduce / output post-process

SGLang #28450 属于第一块：让 top-k 路由表一次变成 DeepEP 最终布局，减少小 kernel launch。

vLLM #44667 属于第三块：让 NVFP4 MoE 权重边反量化边参与 GEMM，避免完整 BF16 中间权重落到 HBM。

后续补充的五篇把主线扩展到三类基础设施：

- SGLang #25820：DeepSeek-V4 mixed FP8+NVFP4 checkpoint 如何接入 MoE FP4 快路径。
- SGLang #28782 / vLLM #45232：把动态 metadata 前移，让 CUDA Graph 能稳定 replay。
- vLLM #45181 / #45424：处理 KV cache 物理 layout 和 CPU->GPU 小 tensor copy 的隐藏同步/布局问题。
