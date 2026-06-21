# MoE 推理加速知识拓扑

## 一张图记住 MoE 推理路径

```text
请求进入 serving engine
  -> scheduler / batch / cuda graph
  -> transformer layer
  -> router 计算 logits
  -> top-k expert 选择
  -> top-k 后处理
       - expert id 映射
       - shared expert 追加
       - padding token mask
       - EPLB / expert location remap
  -> dispatch tokens
       - local gather
       - all-to-all / DeepEP / MoRI / MegaMoE
  -> expert MLP
       - w13 GEMM
       - activation
       - w2 GEMM
       - quant / dequant
  -> combine top-k outputs
  -> attention / next layer / output
```

## 五类常见瓶颈

| 瓶颈 | 典型位置 | 现象 | 常见优化 |
| --- | --- | --- | --- |
| kernel launch | top-k 后处理、padding mask、小 tensor op | profile 里很多很短的 kernel | fuse eager op 到 Triton/CUDA kernel |
| HBM 带宽 | 反量化、copy materialization、transpose/contiguous | 算力没满但显存带宽高 | 边解码边计算、避免中间大 tensor |
| 通信 | EP all-to-all、TP all-reduce、DP+TP 混合通信 | decode 每层有通信 bubble | DeepEP/MoRI/MegaMoE、通信融合、overlap |
| 负载不均 | expert hit 分布倾斜 | 某些 rank 慢，p99 TPOT 高 | EPLB、waterfill、expert placement |
| shape/layout 不匹配 | 量化格式、group size、tile size、TP shard | fallback 到慢路径或额外 repack | shape bucket、专用 layout、kernel config |

## 当前两篇 PR 的定位

| PR | 框架 | 优化层级 | 核心动作 | 记忆点 |
| --- | --- | --- | --- | --- |
| SGLang #28450 | SGLang | top-k 后处理 | append shared expert + DeepEP remap 融合 | top-k 表一次写成最终形态 |
| vLLM #44667 | vLLM | expert GEMM / quant | NVFP4 权重边反量化边 GEMM | FP4 权重不整包解压 |

## 从小白到能读 PR 的最短路径

1. 先看 PR 改了哪个阶段：router、top-k、dispatch、expert GEMM、combine。
2. 再看旧路径有没有大中间 tensor、多个小 kernel、全量 expert 工作。
3. 找入口函数，而不是先读 kernel：入口函数说明什么时候走新路径。
4. 找 correctness test：看新路径是否和 reference bit-identical。
5. 最后看 benchmark：区分 microbench、kernel-only、端到端 serving。

## 四个通用判断题

读任何 MoE 加速 PR 时，先回答：

1. 这个 PR 有没有减少 kernel launch？
2. 有没有避免 HBM 上的大 tensor materialization？
3. 有没有让稀疏 MoE 真正只处理命中的 expert？
4. 有没有改变 expert id / weight 的语义，是否需要 accuracy test？

如果一个 PR 同时命中前两点，通常是高价值优化；如果命中第 4 点，必须重点看 correctness。

