# HF 模型接入推理框架学习路径

这个目录专门记录：从 Hugging Face 拿到一个模型后，如何判断它能不能直接跑，什么时候要写 vLLM / SGLang / TensorRT-LLM 模型代码，以及真正要写哪些代码。

## 当前入口

- [Hugging Face 模型接入 vLLM / SGLang / TensorRT-LLM 手把手流程](hf-to-vllm-sglang-trtllm-step-by-step.md)
- [可视化图解：从 HF checkpoint 到推理框架](../visuals/hf-model-porting-pipeline.html)

## 先记住一句话

```text
模型接入不是“把权重复制进去”。
它是让框架同时理解：
config 结构、模型 forward、权重名字、KV cache、并行切分、量化格式、tokenizer/chat template。
```

## 建议学习顺序

1. 先看 Hugging Face 模型包里有哪些文件。
2. 再学 `config.json` 里的 `architectures` 如何决定框架选哪个模型类。
3. 用一个已有架构模型练习：例如 Qwen2 / Llama。
4. 再做“外部注册一个 wrapper model”的练习。
5. 最后才做真正的新 architecture。
