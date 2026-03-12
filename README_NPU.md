# 昇腾 NPU W8A8 INT8 PPL 评测指南

在昇腾 910B NPU 上对模型进行 BF16 baseline 和 W8A8 INT8 量化的 WikiText-2 PPL 评测。

**评测方式**: `vllm serve` 拉起模型在线推理服务 → `lm-eval` 通过 OpenAI API 评测 PPL。

---

## 文件清单

```
lmeval/
├── tools/
│   ├── npu_ppl_eval.sh               主评测脚本 (serve + eval + stop，统一入口)
│   └── quantize_safetensors_int8.py   RTN INT8 量化 (CPU，无需 GPU/NPU)
├── lm_eval/
│   └── tasks/
│       └── wikitext_local/            离线 WikiText-2 PPL 评测任务
│           ├── wikitext_local.yaml     lm-eval 任务配置
│           ├── preprocess_wikitext_local.py  word_perplexity 计算
│           ├── test-*.parquet          测试数据 (~716KB)
│           ├── train-*.parquet         训练数据 (~6MB)
│           └── validation-*.parquet    验证数据 (~637KB)
└── results/                           评测结果输出目录
```

---

## PanGu V2 MoE 在昇腾 NPU 上的 PPL 评测

### 前提条件

- 已有 omni-infer 容器，且 omni-npu 代码已安装（editable install）
- 容器内可访问 NPU 设备
- 模型权重已准备好（BF16 safetensors 格式）
- 容器内已安装 `lm-eval[api]`（`pip install "lm-eval[api]"`）

### 第 0 步：部署评测工具到容器

将本仓库复制到容器可访问的目录：

```bash
# 方式 1：从蓝区 SCP
cd /home/p00929643/int8/lmeval
tar czf /tmp/lmeval_npu.tar.gz tools/ lm_eval/tasks/wikitext_local/
scp /tmp/lmeval_npu.tar.gz npu-4:/data/p00929643/

# 在 NPU 服务器上解压
ssh npu-4
cd /data/p00929643
mkdir -p lmeval && cd lmeval
tar xzf ../lmeval_npu.tar.gz

# 方式 2：Git clone（需要 GitHub 访问）
git clone git@github.com:KailTes/lmeval.git /data/p00929643/lmeval
```

验证关键文件：

```bash
ls tools/npu_ppl_eval.sh                           # 主脚本
ls lm_eval/tasks/wikitext_local/*.parquet           # 3 个 parquet 文件
ls lm_eval/tasks/wikitext_local/wikitext_local.yaml # 任务配置
```

### 第 1 步：安装 lm-eval（如未安装）

```bash
docker exec -it <container> bash

# 只需 lm-eval[api]，不要安装 llmcompressor（会破坏 torch 版本）
pip install "lm-eval[api]"

# 验证
python3 -c "import lm_eval; print(lm_eval.__version__)"
```

### 第 2 步：量化模型为 INT8 (RTN W8A8)

```bash
# 纯 CPU 量化，不需要 NPU
cd /data/p00929643/lmeval
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python3 tools/quantize_safetensors_int8.py \
    --model /data/weights/pangu_v2/21B/iter_0213000 \
    --output /data/models/PanguV2MoE-21B-W8A8

# PanGu V2 MoE 默认 skip patterns: embed, kv_b_proj, lm_head, shared_head.head
```

量化后 `config.json` 会自动写入 `quantization_config`，vllm 加载时自动识别 INT8。

### 第 3 步：评测 BF16 PPL

```bash
# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=1           # 使用哪些 NPU 卡
export TP_SIZE=1                              # 张量并行数
export SERVE_PORT=8001                        # 端口（避免和已有服务冲突）
export MAX_MODEL_LEN=2048                     # 最大序列长度
export VLLM_USE_V1=1                          # V1 引擎

# 多卡示例（4卡 EP）:
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# export TP_SIZE=4

cd /data/p00929643/lmeval

# 3a. 启动 BF16 模型服务
bash tools/npu_ppl_eval.sh serve /data/weights/pangu_v2/21B/iter_0213000

# 3b. 运行 PPL 评测
bash tools/npu_ppl_eval.sh eval /data/weights/pangu_v2/21B/iter_0213000

# 3c. 停止服务
bash tools/npu_ppl_eval.sh stop
```

> `npu_ppl_eval.sh` 会自动检测 `model_type=deepseek_v3` 并设置 `VLLM_PLUGINS=omni-npu,omni_npu_patches,omni_custom_models`。

### 第 4 步：评测 INT8 PPL

```bash
# 4a. 启动 INT8 模型服务
bash tools/npu_ppl_eval.sh serve /data/models/PanguV2MoE-21B-W8A8

# 4b. 运行 PPL 评测
bash tools/npu_ppl_eval.sh eval /data/models/PanguV2MoE-21B-W8A8

# 4c. 停止服务
bash tools/npu_ppl_eval.sh stop
```

### 第 5 步：对比结果

```bash
# 查看结果
find results/ -name "results.json" -exec echo "=== {} ===" \; -exec python3 -c "
import json, sys
with open('{}') as f:
    data = json.load(f)
for task, metrics in data.get('results', {}).items():
    print(f\"  word_perplexity: {metrics.get('word_perplexity,none', 'N/A')}\")
    print(f\"  bits_per_byte:   {metrics.get('bits_per_byte,none', 'N/A')}\")
" \;
```

关注 `word_perplexity` 指标，INT8 相比 BF16 的劣化通常在 1-2% 以内。

### 一键流程

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
export TP_SIZE=1
export SERVE_PORT=8001
cd /data/p00929643/lmeval

# 方式 1：分步执行
bash tools/npu_ppl_eval.sh run_bf16 /data/weights/pangu_v2/21B/iter_0213000    # BF16
bash tools/npu_ppl_eval.sh run_int8 /data/weights/pangu_v2/21B/iter_0213000    # INT8 (需先量化)

# 方式 2：完整对比 (BF16 → 量化 → INT8，全自动)
bash tools/npu_ppl_eval.sh compare /data/weights/pangu_v2/21B/iter_0213000

# 方式 3：启用 INT8 KV cache 量化 (可选，进一步节省显存)
KV_CACHE_DTYPE=int8 bash tools/npu_ppl_eval.sh run_bf16 /data/weights/pangu_v2/21B/iter_0213000
KV_CACHE_DTYPE=int8 bash tools/npu_ppl_eval.sh run_int8 /data/weights/pangu_v2/21B/iter_0213000
```

---

## 通用模型 PPL 评测

本工具不仅支持 PanGu V2 MoE，也支持其他 vllm 兼容模型：

```bash
cd /data/p00929643/lmeval

# 量化
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python3 tools/quantize_safetensors_int8.py \
    --model /models/Qwen3-0.6B \
    --output /data/models/Qwen3-0.6B-RTN-W8A8

# BF16 评测
bash tools/npu_ppl_eval.sh run_bf16 /models/Qwen3-0.6B

# INT8 评测
bash tools/npu_ppl_eval.sh run_int8 /models/Qwen3-0.6B /data/models/Qwen3-0.6B-RTN-W8A8
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | NPU 设备编号 |
| `TP_SIZE` | `1` | tensor parallel size |
| `SERVE_PORT` | `8000` | vllm serve 端口 |
| `MAX_MODEL_LEN` | `2048` | 最大序列长度 |
| `DTYPE` | `bfloat16` | 模型精度 |
| `VLLM_PLUGINS` | 自动检测 | vllm 插件列表 (PanGu 自动设置) |
| `VLLM_USE_V1` | `0` (PanGu 自动设为 `1`) | vllm V1 引擎开关 |
| `KV_CACHE_DTYPE` | `auto` | KV cache 精度 (`auto`=不量化, `int8`=INT8 量化; 昇腾不支持 FP8) |
| `EXTRA_SERVE_ARGS` | 空 | 额外 vllm serve 参数 |
| `QUANT_OUTPUT_BASE` | 模型所在目录 | 量化输出的父目录 |

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `400 Bad Request: maximum context length is 63` | `max-model-len` 太小 | 设 `MAX_MODEL_LEN=2048` 或更大 |
| `Parquet magic bytes not found` | parquet 文件不完整或损坏 | 重新复制 `wikitext_local/` 下全部 3 个 parquet |
| `ERR99999 UNKNOWN application exception` | lm-eval 客户端加载了 torch_npu | 脚本已自动设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| vllm serve 启动后没有响应 | 启动较慢 | 等待最多 240 秒，检查 `vllm_serve.log` |
| `npu_moe_init_routing` crash | warmup 阶段 MoE 路由不支持 dummy data | 确保 `VLLM_USE_V1=1` |
| NPU 设备被占用 | 其他进程占用了指定 NPU | `npu-smi info` 查看，换一张空闲卡 |
| `fatal: not a git repository` | lm-eval 尝试记录 git hash | 无害警告，可忽略 |
| 端口已被占用 | 其他 vllm 实例正在运行 | 先 `stop`，或换 `SERVE_PORT` |

## 验证结果 (PanGu V2 MoE Dummy, WikiText-2, 单卡 910B1)

| 配置 | bits_per_byte | word_perplexity |
|------|---------------|-----------------|
| BF16 | 33.5685 | 4.43e53 |
| BF16 + INT8 KV cache | 33.5685 | 4.43e53 |
| W8A8 INT8 | 33.5598 | 4.29e53 |
| W8A8 INT8 + INT8 KV cache | 33.5598 | 4.29e53 |

> 数值巨大是因为使用了随机权重，仅验证管线正确性。真实权重的 PPL 应在合理范围内。
> 昇腾 910B 不支持 FP8 KV cache（`--kv-cache-dtype fp8` 会报 `DT_UINT8` 不支持），
> 但原生支持 INT8 KV cache（`--kv-cache-dtype int8`，需自动 patch vllm CLI）。
