# GitHub PR 来源索引

记录本仓库笔记引用过的 PR，方便后续复查状态。

## 2026-06 MoE 推理加速

| PR | 仓库 | 主题 | 本地笔记 |
| --- | --- | --- | --- |
| [#28450](https://github.com/sgl-project/sglang/pull/28450) | SGLang | Fuse shared-expert append + DeepEP remap | [笔记](../01-pr-notes/2026-06/sglang-28450-fused-append-remap-deepep.md) |
| [#44667](https://github.com/vllm-project/vllm/pull/44667) | vLLM | Fuse NVFP4 dequantization with MoE compute | [笔记](../01-pr-notes/2026-06/vllm-44667-nvfp4-moe-fused-dequant.md) |

## 同主题后续可读

| PR | 仓库 | 主题 | 状态 |
| --- | --- | --- | --- |
| [#28084](https://github.com/sgl-project/sglang/pull/28084) | SGLang | Fuse top-k padded-token masking | 待精读 |
| [#46117](https://github.com/vllm-project/vllm/pull/46117) | vLLM | ROCm MXFP8 grouped-MoE GEMM | 待精读 |
| [#43219](https://github.com/vllm-project/vllm/pull/43219) | vLLM | async EPLB default | 待精读 |
| [#43339](https://github.com/vllm-project/vllm/pull/43339) | vLLM | EPLB for DeepSeek V4 Mega-MoE | 待精读 |

## 记录规范

每新增一篇 PR 笔记，应同时更新：

1. `01-pr-notes/<yyyy-mm>/README.md`
2. `sources/github-pr-index.md`
3. 如果抽象出新模式，更新 `02-patterns/moe-optimization-patterns.md`
4. 如果影响迁移方法，更新 `04-model-porting/moe-porting-checklist.md`

