# Hugging Face 模型接入 vLLM / SGLang / TensorRT-LLM 手把手流程

可视化图解：[从 HF checkpoint 到推理框架](../visuals/hf-model-porting-pipeline.html)

参考来源：

- vLLM 官方模型注册文档：[Registering a Model](https://docs.vllm.ai/en/stable/contributing/model/registration.html)
- vLLM 官方模型实现教程：[Basic Model](https://docs.vllm.ai/en/latest/contributing/model/basic/)
- vLLM 支持模型说明：[Supported Models](https://docs.vllm.ai/en/stable/models/supported_models.html)
- SGLang 本地官方文档：`sglang/docs/supported_models/support_new_models.md`
- TensorRT-LLM 官方 workflow：[TensorRT LLM Checkpoint](https://nvidia.github.io/TensorRT-LLM/architecture/checkpoint.html)
- TensorRT-LLM 官方模型定义：[Model Definition](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html)

## 0. 先纠正一个直觉

你从 Hugging Face 下载的不是“一个可以直接被所有框架执行的模型”。它更像一个包：

```text
config.json
tokenizer.json / tokenizer.model
tokenizer_config.json
special_tokens_map.json
generation_config.json
model-00001-of-000xx.safetensors
model.safetensors.index.json
可能还有 processor_config.json / image_processor_config.json
可能还有 quantization_config / hf_quant_config.json
可能还有 remote code: modeling_xxx.py
```

推理框架要做的是把这些信息变成自己的高性能执行路径：

```text
HF config + HF weights
  -> 框架模型类
  -> 框架 attention / MLP / MoE / logits processor
  -> 权重名字映射和切分
  -> KV cache 管理
  -> sampling / serving
```

所以“接入模型”不是只复制权重。真正工作是让框架理解这个模型的结构和权重。

## 1. 第一问：你到底要不要写代码？

先打开 Hugging Face 模型目录里的 `config.json`，找这个字段：

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "hidden_size": 3584,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "num_hidden_layers": 28
}
```

然后用下面这张决策表。

| 情况 | 你要做什么 | 例子 |
| --- | --- | --- |
| `architectures` 已经在框架支持列表里 | 直接跑，不写模型代码 | Qwen2ForCausalLM、LlamaForCausalLM |
| 模型结构和已有模型完全一样，只是 architecture 名不同 | 注册 alias 或改本地 config 测试 | `MyQwen2ForCausalLM` 实际就是 Qwen2 |
| 结构很像已有模型，但多几个 config 字段或小改动 | 复制最接近的模型文件，改少量模块 | Qwen 变体、RoPE scaling 变体 |
| attention / MoE / state space / VLM projector 有新结构 | 需要实现模型代码，可能还要 kernel | 新 attention、新 MoE routing、新视觉塔 |
| 想进 TensorRT-LLM engine | 通常需要 checkpoint convert + engine build | HF -> TRT-LLM checkpoint -> TensorRT engine |

小白先记：

```text
先查 architecture 是否已支持。
能复用就不要写新模型。
必须写时，从最像的模型复制改，不要从空白文件开始。
```

## 2. Hugging Face 模型包里每个文件管什么

| 文件 | 框架为什么关心 |
| --- | --- |
| `config.json` | 决定 architecture、层数、hidden size、head 数、RoPE、MoE、quant 信息 |
| `model.safetensors.index.json` | 告诉 loader 每个权重 tensor 在哪个 shard |
| `*.safetensors` | 真正的权重 |
| `tokenizer.json` / `tokenizer.model` | prompt 文本转 token id |
| `tokenizer_config.json` | chat template、special token 配置 |
| `generation_config.json` | 默认温度、eos、max length 等生成参数 |
| `processor_config.json` | VLM / audio 模型的 processor 参数 |
| `image_processor_config.json` | 图像预处理参数 |
| `hf_quant_config.json` | 一些 ModelOpt / FP4 / FP8 checkpoint 的量化细节 |
| `modeling_xxx.py` | Hugging Face remote code，框架不一定能直接高性能运行 |

## 3. 标准总流程

```text
Step 1: 下载模型
Step 2: 读 config.json，识别 architecture
Step 3: 找框架是否已有同类模型
Step 4: 决定复用、注册 alias、还是写新模型类
Step 5: 实现模型 forward
Step 6: 实现/适配 load_weights
Step 7: 注册模型
Step 8: 跑 HF reference 对齐
Step 9: 跑框架单 batch correctness
Step 10: 跑 serving、benchmark、accuracy
```

你每次接入新模型都按这 10 步走，不要直接冲进 kernel。

## 4. Step 1：下载并检查模型

先用 Hugging Face CLI 或 Python 下载。示例：

```bash
huggingface-cli download Qwen/Qwen2-1.5B-Instruct --local-dir ./ckpt/qwen2-1.5b
```

检查文件：

```bash
ls ./ckpt/qwen2-1.5b
```

看 `config.json`：

```bash
python - <<'PY'
import json
p = "./ckpt/qwen2-1.5b/config.json"
cfg = json.load(open(p, encoding="utf-8"))
for k in [
    "architectures",
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "rope_theta",
    "sliding_window",
    "tie_word_embeddings",
    "quantization_config",
]:
    print(k, "=", cfg.get(k))
PY
```

你要把输出手动整理成一张表：

```text
architecture:
decoder layers:
hidden size:
attention heads:
kv heads:
head dim:
MLP type:
activation:
RoPE:
MoE:
quant:
VLM/audio:
```

这张表决定你后面改哪个模型文件。

## 5. Step 2：找最像的已有模型

新手不要问“我要怎么从零写”。先问：

```text
这个模型最像 Llama、Qwen2、Qwen3、Mixtral、DeepSeek、Gemma、Mistral 里的哪一个？
```

在 vLLM 里看：

```text
vllm/vllm/model_executor/models/
```

在 SGLang 里看：

```text
sglang/python/sglang/srt/models/
```

比如 Qwen2：

```text
vLLM:   vllm/model_executor/models/qwen2.py
SGLang: python/sglang/srt/models/qwen2.py
```

你要比较的是结构，不是名字：

```text
embedding
decoder layer
self attention
QKV projection 是分开的还是 fused 的
RoPE 位置
MLP 是 gate/up/down 还是别的
norm 是 RMSNorm 还是 LayerNorm
lm_head 是否 tied embedding
```

## 6. vLLM 接入路线

### 6.1 vLLM 怎么根据 HF config 找模型类

vLLM 读 `config.json` 的 `architectures`，然后查：

```text
vllm/model_executor/models/registry.py
```

里面有类似：

```python
_TEXT_GENERATION_MODELS = {
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
}
```

意思是：

```text
HF config 里 architecture = Qwen2ForCausalLM
  -> import vllm.model_executor.models.qwen2
  -> 找 Qwen2ForCausalLM 类
```

如果是上游内置支持，你要做：

```text
1. 新增 vllm/model_executor/models/my_model.py
2. 在 registry.py 的 _VLLM_MODELS 里加 architecture -> 文件/类 映射
3. 更新 supported models 文档和测试
```

如果是你自己的外部实验，官方推荐用 plugin / out-of-tree 注册：

```python
def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "MyModelForCausalLM",
        "my_package.my_model:MyModelForCausalLM",
    )
```

用字符串懒加载可以避免 import 时提前初始化 CUDA。

### 6.2 vLLM 模型文件最小骨架

下面是读代码时要认的骨架，不是完整可跑模型。

```python
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.attention import Attention
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.models.utils import AutoWeightsLoader
```

#### 第 1 行思路：定义 MLP

```python
class MyMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, quant_config=None, prefix=""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        # 这里通常接 SiluAndMul
        x = self.down_proj(gate_up)[0]
        return x
```

你要注意 `prefix`。vLLM 官方教程强调，vLLM modules 要带 `prefix`，它用于 attention state、量化匹配和权重名定位。

#### 第 2 行思路：定义 Attention

```python
class MyAttention(nn.Module):
    def __init__(self, config, cache_config, quant_config=None, prefix=""):
        super().__init__()
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            config.hidden_size // config.num_attention_heads,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.attn = Attention(
            num_heads=config.num_attention_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            scale=(config.hidden_size // config.num_attention_heads) ** -0.5,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        # split q/k/v, apply rope, call self.attn, then o_proj
        ...
```

真实代码要补 RoPE。你可以照 `qwen2.py` 或 `llama.py` 改。

#### 第 3 行思路：定义 DecoderLayer

```python
class MyDecoderLayer(nn.Module):
    def __init__(self, config, cache_config=None, quant_config=None, prefix=""):
        super().__init__()
        self.self_attn = MyAttention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = MyMLP(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        ...
```

#### 第 4 行思路：定义主干模型

```python
class MyModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.layers = nn.ModuleList([
            MyDecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

真实 vLLM 代码还要处理 pipeline parallel 的 `PPMissingLayer` 和 `IntermediateTensors`。小白第一遍先把单卡逻辑看懂。

#### 第 5 行思路：定义 ForCausalLM 外壳

```python
class MyModelForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.model = MyModel(vllm_config=vllm_config, prefix="model")
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix="lm_head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states, sampling_metadata):
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    def sample(self, logits, sampling_metadata):
        return self.sampler(logits, sampling_metadata)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
```

### 6.3 vLLM 最容易错的地方

| 坑 | 现象 |
| --- | --- |
| 忘记 `prefix` | attention state / 量化 / 权重加载对不上 |
| HF 权重是 `q_proj/k_proj/v_proj`，框架层是 `qkv_proj` | 需要在 `load_weights` 做 stacked mapping |
| `gate_proj/up_proj` 合并成 `gate_up_proj` | MLP 权重要按 shard_id 加载 |
| `tie_word_embeddings=true` | `lm_head` 可能复用 embedding |
| TP/PP 没处理 | 单卡能跑，多卡错 |
| RoPE scaling 字段没读 | 长上下文输出错 |
| sliding window / attention type 漏掉 | 推理结果或性能错 |
| 量化 scale 名字不匹配 | 权重加载成功但数值错 |

## 7. SGLang 接入路线

### 7.1 SGLang 怎么找到模型类

SGLang 的 registry 会扫描：

```text
python/sglang/srt/models/
```

每个模型文件末尾通常有：

```python
EntryClass = Qwen2ForCausalLM
```

或者：

```python
EntryClass = [AForCausalLM, BForCausalLM]
```

registry 会把 `EntryClass.__name__` 注册成 architecture 名。也就是说：

```text
HF config architectures = ["Qwen2ForCausalLM"]
  -> SGLang registry 找 Qwen2ForCausalLM
```

SGLang 也有 Transformers fallback，但高性能支持通常还是写 SGLang native model。

### 7.2 SGLang 模型文件和 vLLM 的差异

SGLang 很多模型从 vLLM 改来，但要替换成 SGLang runtime 组件。

| vLLM | SGLang |
| --- | --- |
| `Attention` | `RadixAttention` |
| `LogitsProcessor(config.vocab_size)` | `LogitsProcessor(config)` |
| `input_ids, positions, intermediate_tensors` | `input_ids, positions, forward_batch` |
| `IntermediateTensors` | `PPProxyTensors` |
| vLLM layers | SGLang `srt.layers` |
| registry dict | 文件末尾 `EntryClass` |

SGLang 官方文档也明确说，从 vLLM port 到 SGLang 时，要替换 attention、logits processor、forward 签名，并添加 `EntryClass`。

### 7.3 SGLang 最小骨架

```python
import torch
from torch import nn

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.layers.quantization.base_config import QuantizationConfig
```

#### 第 1 行思路：Attention 要拿 `forward_batch`

```python
class MyAttention(nn.Module):
    def __init__(self, config, layer_id: int, quant_config=None, prefix=""):
        super().__init__()
        self.qkv_proj = QKVParallelLinear(...)
        self.o_proj = RowParallelLinear(...)
        self.attn = RadixAttention(
            num_heads=...,
            head_dim=...,
            scaling=...,
            num_kv_heads=...,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions, hidden_states, forward_batch: ForwardBatch):
        qkv, _ = self.qkv_proj(hidden_states)
        # split q/k/v, apply rope
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output
```

和 vLLM 最大差异就是：

```text
SGLang attention 需要 forward_batch
```

这里面有 batch、KV cache、decode/extend 等 runtime metadata。

#### 第 2 行思路：ForCausalLM forward 返回 logits processor output

```python
class MyModelForCausalLM(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig | None = None, prefix=""):
        super().__init__()
        self.config = config
        self.model = MyModel(config, quant_config=quant_config, prefix="model")
        self.lm_head = ParallelLMHead(...)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )
```

#### 第 3 行思路：末尾注册

```python
EntryClass = MyModelForCausalLM
```

### 7.4 SGLang 权重加载

如果你的层名字和 HF 一样，简单 loader 可以工作。只要合并了 QKV 或 gate/up，就需要 mapping。

典型 mapping：

```python
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]
```

这个表的意思是：

```text
HF 里三个 tensor:
  q_proj.weight
  k_proj.weight
  v_proj.weight

框架里一个 fused 参数:
  qkv_proj.weight

加载时按 shard_id 分别写进去。
```

### 7.5 SGLang 测试命令

SGLang 官方文档建议先拿 Hugging Face reference 对比：

```bash
python3 scripts/playground/reference_hf.py --model-path [new model] --model-type text
```

再跑 SGLang 单 batch correctness：

```bash
python3 -m sglang.bench_one_batch --correct --model [new model]
```

如果要进上游，还要把模型加入测试列表，并报告 accuracy / benchmark。

## 8. TensorRT-LLM 接入路线

TensorRT-LLM 和 vLLM/SGLang 的心智模型不同。

vLLM/SGLang 通常是：

```text
HF safetensors
  -> Python model class
  -> runtime 直接加载权重
```

TensorRT-LLM 典型 engine 路线是：

```text
HF / NeMo / JAX / DeepSpeed weights
  -> convert
  -> TensorRT-LLM checkpoint
  -> build
  -> TensorRT engine
  -> ModelRunner / serving
```

NVIDIA 官方 checkpoint workflow 明确把它分成三步：

```text
Convert weights -> Build TensorRT engines -> Load engines and evaluate
```

### 8.1 已支持模型

如果架构已支持，你通常做：

```bash
python convert_checkpoint.py \
  --model_dir ./hf_model \
  --output_dir ./trtllm_ckpt \
  --dtype bfloat16

trtllm-build \
  --checkpoint_dir ./trtllm_ckpt \
  --output_dir ./engine \
  --max_batch_size 8 \
  --max_input_len 4096 \
  --max_seq_len 8192

python run.py \
  --engine_dir ./engine \
  --tokenizer_dir ./hf_model
```

具体脚本名和参数会随 TensorRT-LLM 版本和模型变化。现在官方也在推进更统一的 LLM API / `trtllm-serve` 路线，所以接入前一定看当前模型目录和官方 workflow。

### 8.2 新 architecture

如果 TensorRT-LLM 里没有这个 architecture，你要做的更多：

1. 定义 TensorRT-LLM model graph。
2. 写 HF -> TensorRT-LLM checkpoint 的权重转换。
3. 处理 tensor parallel / pipeline parallel 权重切分。
4. 处理 attention plugin、GEMM plugin、MoE plugin、quant plugin。
5. build engine。
6. runner 对齐 HF reference。

TensorRT-LLM 的难点是：它不是只写 PyTorch forward，而是要把网络定义成 TensorRT 可编译的 graph，并处理 engine build 的 shape/profile。

## 9. 最小练习路线：不要一上来做新架构

### 练习 A：只跑已支持模型

目标：理解 HF config 如何被框架识别。

```text
Qwen/Qwen2-1.5B-Instruct
architectures = Qwen2ForCausalLM
vLLM/SGLang 都已有实现
```

你要做：

1. 下载模型。
2. 打印 config。
3. 找 vLLM/SGLang 对应 qwen2.py。
4. 跑一条 prompt。

### 练习 B：做一个 wrapper 模型

目标：理解 registry。

做法：

```text
继承 Qwen2ForCausalLM
不改结构
只在 logits 上加一个很小的 bias
注册成 MyQwen2ForCausalLM
把本地 config.json 的 architectures 改成 MyQwen2ForCausalLM
```

这样你不用先写 attention / MLP，就能理解：

```text
architectures -> registry -> model class
```

### 练习 C：复制 Qwen2 改一个小结构

目标：理解权重加载和 forward。

比如只加一个 config 字段，或者改 activation 的判断。不要先碰新 attention。

### 练习 D：真正 port 新模型

目标：完整流程。

你要做：

1. 画结构图。
2. 找最近参考模型。
3. 写 model file。
4. 写 weight mapping。
5. 注册。
6. 对齐 HF logits。
7. 多卡/量化/长上下文测试。

## 10. correctness 怎么做

不要只看生成文本。生成文本太容易受 sampling 影响。

最低要求：

```text
同一个 prompt
同一个 tokenizer
同一个 dtype
同一个 max_new_tokens
temperature=0
比较 prefill logits / first token logits
```

建议分层检查：

| 层级 | 检查 |
| --- | --- |
| tokenizer | 同一段文本 token id 完全一样 |
| embedding | 第一个 token embedding shape 对 |
| single layer | hidden states 范围正常 |
| logits | HF vs framework close |
| greedy output | temperature=0 输出一致或高度接近 |
| serving | OpenAI API 请求能跑 |
| benchmark | TTFT / TPOT / throughput |
| accuracy | GSM8K / MMLU / MMMU 等 |

## 11. 新手最常见的 20 个问题

### 1. 为什么 HF `AutoModelForCausalLM` 能跑，vLLM/SGLang 还要写模型？

HF 是通用 PyTorch 实现，目标是正确和训练/推理都能用。vLLM/SGLang 要高吞吐 serving，需要自己的 attention、KV cache、continuous batching、TP/PP、quantization，所以不能直接套所有 HF forward。

### 2. `architectures` 和 `model_type` 有什么区别？

`architectures` 更直接决定模型类，比如 `Qwen2ForCausalLM`。`model_type` 更像配置族名，比如 `qwen2`。框架通常先看 `architectures`。

### 3. tokenizer 要写代码吗？

多数文本模型不需要。框架直接用 HF tokenizer。VLM/audio 模型通常要处理 processor、多模态 token、image/audio feature。

### 4. chat template 是模型代码吗？

不是。它影响 prompt 如何拼成文本和 special tokens，但不改变模型 forward。错误 chat template 会让模型输出质量很差，看起来像模型实现错。

### 5. safetensors 要转换吗？

vLLM/SGLang 通常直接读 HF safetensors，只要权重名字能映射。TensorRT-LLM engine 路线通常要先转成 TensorRT-LLM checkpoint，再 build engine。

### 6. 为什么 QKV 经常被合并？

为了更少 kernel 和更好 GEMM。HF 里可能是 `q_proj/k_proj/v_proj`，框架里为了性能合成 `qkv_proj`。

### 7. 为什么 gate/up 也合并？

SwiGLU MLP 里 gate 和 up 通常同输入，可以合成一次 column parallel GEMM，再做 activation。

### 8. 什么是 TP？

Tensor parallel，把一个大权重矩阵切到多张 GPU。模型代码里的 parallel linear 就是在处理这个。

### 9. 什么是 PP？

Pipeline parallel，把不同层放到不同 GPU/rank。模型代码要处理“我是不是第一层/最后一层”。

### 10. 什么是 KV cache？

decode 时过去 token 的 key/value 缓存。高性能推理框架最核心的能力之一就是高效管理 KV cache。

### 11. 为什么 attention 不能直接用 `torch.nn.MultiheadAttention`？

serving 需要 paged KV cache、decode/prefill 分离、continuous batching、flash attention 等。普通 PyTorch attention 不适合。

### 12. 新 RoPE scaling 会影响什么？

影响 position embedding。如果没读对，短文本可能看不出来，长文本会明显错。

### 13. MoE 模型多做什么？

还要实现 router、top-k、expert 并行、dispatch/combine、shared expert、expert load balance、MoE quant。

### 14. 量化模型多做什么？

要读 quant config、scale、zero point、group size、packing layout，并确保 quant method 支持这些权重。

### 15. VLM 多做什么？

要处理 processor、图像预处理、视觉塔、projector、图像 token 占位、embedding 拼接和多模态 cache。

### 16. 怎么判断是代码错还是权重名字错？

先看 missing/unexpected weights；再打印某几个关键权重 shape；然后比较第一层输出。很多问题其实是 weight mapping 错。

### 17. 为什么单卡能跑，多卡错？

通常是 TP/PP 切分、rank 本地 head 数、KV head replication、lm_head tied weight 没处理好。

### 18. 为什么输出文本不一样？

先关 sampling：`temperature=0`。再比较 logits。如果 logits 一开始就差很多，是模型/权重/position/tokenizer 问题。

### 19. 为什么框架支持 Transformers fallback，还要 native model？

fallback 是为了覆盖更多模型，性能和功能不一定等同 native。生产服务一般希望 native path。

### 20. 我应该先学哪个框架？

先学 vLLM/SGLang 的 native PyTorch 模型接入。等理解 config、forward、load_weights、KV cache 后，再看 TensorRT-LLM 的 convert/build/engine。

## 12. 你真正写 PR 时的 checklist

```text
[ ] config.json 已分析，字段表已写
[ ] tokenizer/chat template 已验证
[ ] 找到最接近的已有模型
[ ] 模型文件实现 forward
[ ] attention 使用框架 runtime attention
[ ] MLP/Norm/Embedding 使用框架 layer
[ ] QKV/gate_up 权重 mapping 正确
[ ] tied embedding 处理
[ ] TP/PP 基本支持
[ ] quant config 不支持时明确 fallback 或报错
[ ] HF reference logits 对齐
[ ] 框架 correct 单 batch 通过
[ ] serving API 能跑
[ ] benchmark 和 accuracy 记录
[ ] docs / supported model list / tests 更新
```

## 13. 一句话总复盘

```text
Hugging Face 给你模型定义和权重。
vLLM/SGLang 要你把模型改成 serving runtime 能高效执行的 inference-only 代码。
TensorRT-LLM 要你把模型转成可 build 的 TensorRT-LLM checkpoint 和 engine。
```

先学会看 `config.json -> architectures -> registry -> model file -> load_weights` 这条链路，后面的 MoE、量化、VLM 都是在这条链路上增加复杂度。
