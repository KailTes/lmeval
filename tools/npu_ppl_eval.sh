#!/bin/bash
# ============================================================
# 昇腾 NPU W8A8 INT8 PPL 评测全流程
#
# 基于 lm-evaluation-harness (lmeval fork)，通过 vllm serve +
# OpenAI API 进行离线 WikiText-2 PPL 评测。
#
# 用法:
#   bash tools/npu_ppl_eval.sh serve        /path/to/model  — 启动 vllm serve
#   bash tools/npu_ppl_eval.sh eval         /path/to/model  — PPL 评测 (需先 serve)
#   bash tools/npu_ppl_eval.sh stop                         — 停止 vllm serve
#   bash tools/npu_ppl_eval.sh quantize_rtn /path/to/model  — RTN W8A8 INT8 量化 (CPU)
#   bash tools/npu_ppl_eval.sh run_bf16     /path/to/model  — 一键 BF16: serve + eval + stop
#   bash tools/npu_ppl_eval.sh run_int8     /path/to/model  — 一键 INT8: serve + eval + stop
#   bash tools/npu_ppl_eval.sh compare      /path/to/model  — 完整对比: BF16 + 量化 + INT8
#
# 环境变量:
#   ASCEND_RT_VISIBLE_DEVICES  — NPU 设备 (默认: 0)
#   TP_SIZE                    — tensor parallel size (默认: 1)
#   SERVE_PORT                 — vllm serve 端口 (默认: 8000)
#   MAX_MODEL_LEN              — 最大序列长度 (默认: 2048)
#   DTYPE                      — 模型精度 (默认: bfloat16)
#   VLLM_PLUGINS               — vllm 插件 (PanGu 自动设置)
#   EXTRA_SERVE_ARGS           — 额外 vllm serve 参数
# ============================================================

set -uo pipefail

# 定位项目根目录 (lmeval repo root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${SCRIPT_DIR}"
TASK_DIR="${PROJECT_ROOT}/lm_eval/tasks/wikitext_local"

SERVE_PORT="${SERVE_PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
API_BASE="${API_BASE:-http://localhost:${SERVE_PORT}}"

# 确保本地请求不走代理
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1"
export NO_PROXY="${no_proxy}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---- 工具函数 ----

# 从模型路径推导输出路径
resolve_paths() {
    MODEL_PATH="$1"
    MODEL_NAME="$(basename "${MODEL_PATH}")"
    local out_base="${QUANT_OUTPUT_BASE:-$(dirname "${MODEL_PATH}")}"
    QUANT_OUTPUT_RTN="${out_base}/${MODEL_NAME}-RTN-W8A8"
    RESULTS_DIR="${PROJECT_ROOT}/results"
    RESULTS_PATH="${RESULTS_DIR}/${MODEL_NAME}"
}

# 自动查找 CANN set_env.sh
source_cann() {
    for _cann in /usr/local/Ascend/cann-*/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
        [ -f "$_cann" ] && { source "$_cann" 2>/dev/null; return; }
    done
}

# 自动检测模型类型
detect_model_type() {
    python3 -c "
import json, os
cfg = os.path.join('$1', 'config.json')
with open(cfg) as f:
    print(json.load(f).get('model_type', 'auto').lower())
" 2>/dev/null || echo "auto"
}

# 修正 yaml 中的数据路径占位符 (只执行一次)
fix_yaml_path() {
    local yaml_file="${TASK_DIR}/wikitext_local.yaml"
    if grep -q '__TASK_DIR__' "${yaml_file}"; then
        sed -i "s|__TASK_DIR__|${TASK_DIR}|g" "${yaml_file}"
        info "Fixed dataset path → ${TASK_DIR}/"
    fi
}

# 等待 vllm serve 就绪
wait_for_serve() {
    info "Waiting for vllm serve to be ready on port ${SERVE_PORT} ..."
    for i in $(seq 1 120); do
        if curl --noproxy '*' -s "${API_BASE}/v1/models" > /dev/null 2>&1; then
            info "vllm serve is ready!"
            return 0
        fi
        sleep 2
    done
    error "vllm serve failed to start within 240 seconds. Check ${PROJECT_ROOT}/vllm_serve.log"
}

# ---- serve ----
do_serve() {
    [ -z "${1:-}" ] && error "Usage: npu_ppl_eval.sh serve /path/to/model"
    local model_path="$1"

    source_cann

    local model_type
    model_type=$(detect_model_type "${model_path}")

    export OMNI_NPU_PATCHES_DIR="${model_type}"
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
    export VLLM_USE_V1="${VLLM_USE_V1:-0}"

    # 检测服务是否已在运行
    if curl --noproxy '*' -s "${API_BASE}/v1/models" > /dev/null 2>&1; then
        info "Service already running on port ${SERVE_PORT}, skipping launch"
        return 0
    fi

    info "=== Starting vllm serve ==="
    info "Model: ${model_path}"
    info "Model type: ${model_type}"
    info "Port: ${SERVE_PORT}"
    info "TP size: ${TP_SIZE}"

    # PanGu V2 MoE (model_type=deepseek_v3) 需要特殊环境变量
    if [ "${model_type}" = "deepseek_v3" ]; then
        export VLLM_PLUGINS="${VLLM_PLUGINS:-omni-npu,omni_npu_patches,omni_custom_models}"
        export OMNI_NPU_PATCHES_DIR="deepseek_v3"
        export VLLM_USE_V1="${VLLM_USE_V1:-1}"
        info "PanGu V2 MoE detected → VLLM_PLUGINS=${VLLM_PLUGINS}, VLLM_USE_V1=${VLLM_USE_V1}"
    fi

    [ -n "${VLLM_PLUGINS:-}" ] && info "VLLM_PLUGINS: ${VLLM_PLUGINS}"

    local max_model_len="${MAX_MODEL_LEN:-2048}"

    python3 -m vllm.entrypoints.openai.api_server \
        --model "${model_path}" \
        --dtype "${DTYPE:-bfloat16}" \
        --gpu-memory-utilization 0.8 \
        --enforce-eager \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-model-len "${max_model_len}" \
        --host 0.0.0.0 \
        --port "${SERVE_PORT}" \
        ${EXTRA_SERVE_ARGS:-} \
        > "${PROJECT_ROOT}/vllm_serve.log" 2>&1 &

    SERVE_PID=$!
    echo "${SERVE_PID}" > "${PROJECT_ROOT}/.serve_pid"
    info "Server started (PID: ${SERVE_PID}), log: ${PROJECT_ROOT}/vllm_serve.log"
    wait_for_serve
}

# ---- stop ----
do_stop() {
    info "=== Stopping vllm serve ==="
    if [ -f "${PROJECT_ROOT}/.serve_pid" ]; then
        local pid
        pid=$(cat "${PROJECT_ROOT}/.serve_pid")
        kill "${pid}" 2>/dev/null || true
        sleep 2
        kill -9 "${pid}" 2>/dev/null || true
        rm -f "${PROJECT_ROOT}/.serve_pid"
    fi
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 2
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "APIServer" 2>/dev/null || true
    sleep 1
    info "vllm serve stopped"
}

# ---- lm-eval PPL 评测 ----
do_lmeval() {
    [ -z "${1:-}" ] && error "Usage: npu_ppl_eval.sh eval /path/to/model [output_dir]"
    local model_path="$1"
    local output_dir="${2:-${PROJECT_ROOT}/results/$(basename "${model_path}")}"

    source_cann
    fix_yaml_path

    # 从服务获取实际注册的模型名
    local served_model
    served_model=$(curl --noproxy '*' -s "${API_BASE}/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "${model_path}")

    info "=== PPL Assessment via API ==="
    info "Model: ${model_path}"
    info "Served model name: ${served_model}"
    info "API: ${API_BASE}/v1/completions"
    info "Output: ${output_dir}"

    mkdir -p "${output_dir}"

    # TORCH_DEVICE_BACKEND_AUTOLOAD=0 防止 lm-eval 客户端误加载 torch_npu
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    HF_DATASETS_OFFLINE=1 \
    no_proxy="*" NO_PROXY="*" \
    python3 -m lm_eval --model local-completions \
        --model_args "model=${served_model},base_url=${API_BASE}/v1/completions,tokenizer_backend=huggingface,tokenizer=${model_path},trust_remote_code=True" \
        --include_path "${PROJECT_ROOT}/lm_eval/tasks" \
        --tasks wikitext_local \
        --batch_size auto \
        --output_path "${output_dir}"

    info "Assessment done → ${output_dir}"
}

# ---- quantize_rtn ----
do_quantize_rtn() {
    [ -z "${1:-}" ] && error "Usage: npu_ppl_eval.sh quantize_rtn /path/to/model [output]"
    resolve_paths "$1"
    local output="${2:-${QUANT_OUTPUT_RTN}}"

    info "=== RTN W8A8 Quantize (CPU, pure safetensors) ==="
    info "Input:  ${MODEL_PATH}"
    info "Output: ${output}"

    if [ -d "${output}" ] && [ -n "$(ls "${output}"/*.safetensors 2>/dev/null)" ]; then
        warn "Already exists: ${output}, skipping"
        return 0
    fi

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    python3 "${TOOLS_DIR}/quantize_safetensors_int8.py" \
        --model "${MODEL_PATH}" --output "${output}"
    info "Quantize done → ${output}"
}

# ---- 一键流程 ----

do_run_bf16() {
    [ -z "${1:-}" ] && error "Usage: npu_ppl_eval.sh run_bf16 /path/to/model"
    resolve_paths "$1"
    info "====== BF16 Pipeline: serve → assess → stop ======"
    do_serve "${MODEL_PATH}"
    do_lmeval "${MODEL_PATH}" "${RESULTS_PATH}-bf16" || true
    do_stop
}

do_run_int8() {
    [ -z "${1:-}" ] && error "Usage: npu_ppl_eval.sh run_int8 /path/to/model [int8_path]"
    resolve_paths "$1"
    local int8_path="${2:-${QUANT_OUTPUT_RTN}}"
    [ -d "${int8_path}" ] || error "INT8 model not found: ${int8_path}. Run 'quantize_rtn' first."
    info "====== INT8 Pipeline: serve → assess → stop ======"
    do_serve "${int8_path}"
    do_lmeval "${int8_path}" "${RESULTS_PATH}-int8" || true
    do_stop
}

do_compare() {
    [ -z "${1:-}" ] && error "Usage: npu_ppl_eval.sh compare /path/to/model"
    resolve_paths "$1"
    info "====== Full Compare: BF16 → Quantize → INT8 ======"

    # BF16
    do_serve "${MODEL_PATH}"
    do_lmeval "${MODEL_PATH}" "${RESULTS_PATH}-bf16" || true
    do_stop

    # 量化
    do_quantize_rtn "${MODEL_PATH}"

    # INT8
    do_serve "${QUANT_OUTPUT_RTN}"
    do_lmeval "${QUANT_OUTPUT_RTN}" "${RESULTS_PATH}-int8" || true
    do_stop

    info "========================================="
    info "All done! Results:"
    info "  BF16: ${RESULTS_PATH}-bf16"
    info "  INT8: ${RESULTS_PATH}-int8"
    info "========================================="
}

# ---- main ----
case "${1:-help}" in
    serve)        do_serve "${2:-}" ;;
    eval)         do_lmeval "${2:-}" "${3:-}" ;;
    stop)         do_stop ;;
    quantize_rtn) do_quantize_rtn "${2:-}" "${3:-}" ;;
    run_bf16)     do_run_bf16 "${2:-}" ;;
    run_int8)     do_run_int8 "${2:-}" "${3:-}" ;;
    compare)      do_compare "${2:-}" ;;
    *)
        echo "昇腾 NPU W8A8 INT8 PPL 评测工具"
        echo ""
        echo "Usage: bash tools/npu_ppl_eval.sh <command> [model_path]"
        echo ""
        echo "Commands:"
        echo "  serve         /path/to/model           启动 vllm serve"
        echo "  eval          /path/to/model [output]  PPL 评测 (需先 serve)"
        echo "  stop                                   停止 vllm serve"
        echo "  quantize_rtn  /path/to/model [output]  RTN W8A8 INT8 量化 (CPU)"
        echo "  run_bf16      /path/to/model           一键 BF16: serve + eval + stop"
        echo "  run_int8      /path/to/model [int8_path] 一键 INT8: serve + eval + stop"
        echo "  compare       /path/to/model           完整对比: BF16 + 量化 + INT8"
        echo ""
        echo "环境变量:"
        echo "  ASCEND_RT_VISIBLE_DEVICES  NPU 设备 (默认: 0)"
        echo "  TP_SIZE                    tensor parallel size (默认: 1)"
        echo "  SERVE_PORT                 vllm serve 端口 (默认: 8000)"
        echo "  MAX_MODEL_LEN              最大序列长度 (默认: 2048)"
        echo "  DTYPE                      模型精度 (默认: bfloat16)"
        echo "  VLLM_PLUGINS               vllm 插件 (PanGu 自动设置)"
        echo "  EXTRA_SERVE_ARGS           额外 vllm serve 参数"
        echo ""
        echo "示例:"
        echo "  # 单卡 BF16 PPL"
        echo "  ASCEND_RT_VISIBLE_DEVICES=1 bash tools/npu_ppl_eval.sh run_bf16 /data/models/PanGu-21B"
        echo ""
        echo "  # 4卡 EP 完整对比"
        echo "  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 bash tools/npu_ppl_eval.sh compare /data/models/PanGu-21B"
        ;;
esac
