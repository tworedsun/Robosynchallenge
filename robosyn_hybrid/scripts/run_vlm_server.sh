#!/usr/bin/env bash
# =============================================================================
# run_vlm_server.sh —— 启动一个 OpenAI 兼容的本地 VLM 服务
#
# 用途：RoboSynChallenge 混合提交流程（robosyn_hybrid/vlm.py）需要一个
#       OpenAI 兼容的 HTTP chat/completions 端点。
#
# ★ 推荐后端：ollama + qwen3vl:8b
#   理由：参考实现的 holder / 按钮定位精度（筒口 3D 误差 3.17 mm）就是在
#   ollama + `qwen3vl:8b` 上实测出来的；本仓库的 `settings.vlm_model`
#   默认值也跟随它。ollama 原生提供 OpenAI 兼容端点，不需要 mmproj。
#
#     ollama pull qwen3vl:8b
#     OLLAMA_HOST=127.0.0.1:11434 ollama serve &
#     export HYBRID_VLM_URL=http://127.0.0.1:11434/v1/chat/completions
#     export HYBRID_VLM_MODEL=qwen3vl:8b
#
#   本脚本也支持直接拉起 ollama：
#     VLM_BACKEND=ollama bash robosyn_hybrid/scripts/run_vlm_server.sh
#
# 备用后端：llama.cpp 的 `llama-server`（需 --mmproj 多模态投影）
#
# 用法：
#   bash robosyn_hybrid/scripts/run_vlm_server.sh
#   MODEL_PATH=/data/models/Qwen3-VL-8B-Q4_K_M.gguf \
#   MMPROJ_PATH=/data/models/mmproj-Qwen3-VL-8B-f16.gguf \
#   VLM_PORT=8080 bash robosyn_hybrid/scripts/run_vlm_server.sh
#
# 可覆盖的环境变量（全部有默认值）：
#   VLM_BACKEND       "llama"（默认）或 "ollama"
#   VLM_OLLAMA_MODEL  ollama 模型名，默认 qwen3vl:8b
#   MODEL_PATH        VLM 主权重（GGUF）路径（仅 llama 后端）
#   MMPROJ_PATH       多模态投影 mmproj 文件路径（仅 llama 后端；显式留空 MMPROJ_PATH= 则不加）
#   VLM_PORT          监听端口，默认 8080
#   VLM_HOST          监听地址，默认 127.0.0.1（只监听本机）
#   VLM_CTX_SIZE      上下文长度，默认 16384
#   VLM_GPU_LAYERS    卸载到 GPU 的层数，默认 99；纯 CPU 可设 0
#   LLAMA_SERVER_BIN  llama-server 可执行文件路径（默认从 PATH 查找）
#   VLM_EXTRA_ARGS   追加给 llama-server 的额外参数（空格分隔）
#   ROBOSYN_HYBRID_ASSETS  资产根目录（默认 <repo>/robosyn_hybrid_assets）
#
# 其他替代方案（若本机已有支持 Qwen-VL 的 vLLM）：
#   python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen3-VL-8B-Instruct \
#     --host 127.0.0.1 --port "${VLM_PORT}" \
#     --limit-mm-per-prompt image=2
# 启动后同样导出本脚本末尾打印的 HYBRID_VLM_URL 即可。
# =============================================================================

set -euo pipefail

# --- 路径解析：脚本位于 <repo>/scripts/，资产默认在 <repo>/robosyn_hybrid_assets ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ASSETS_DIR="${ROBOSYN_HYBRID_ASSETS:-${REPO_ROOT}/robosyn_hybrid_assets}"

# --- 可覆盖参数（清晰默认值） ---
MODEL_PATH="${MODEL_PATH:-${ASSETS_DIR}/vlm/Qwen3-VL-8B-Instruct-Q4_K_M.gguf}"
# MMPROJ_PATH 用 "-"（而非 ":-"）展开：显式设为空串 = 主动跳过 --mmproj。
MMPROJ_PATH="${MMPROJ_PATH-${ASSETS_DIR}/vlm/mmproj-Qwen3-VL-8B-Instruct-f16.gguf}"
VLM_PORT="${VLM_PORT:-8080}"
VLM_HOST="${VLM_HOST:-127.0.0.1}"
VLM_CTX_SIZE="${VLM_CTX_SIZE:-16384}"
VLM_GPU_LAYERS="${VLM_GPU_LAYERS:-99}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-}"
VLM_EXTRA_ARGS="${VLM_EXTRA_ARGS:-}"
# 后端选择：默认 llama.cpp；VLM_BACKEND=ollama 走 ollama（推荐，见文件头）。
VLM_BACKEND="${VLM_BACKEND:-llama}"
# ollama 的模型名。与 settings.py 的 _DEFAULT_VLM_MODEL 保持一致。
VLM_OLLAMA_MODEL="${VLM_OLLAMA_MODEL:-qwen3vl:8b}"
# ollama 的惯例端口是 11434。若用户没有显式改过 VLM_PORT（仍是默认 8080），
# 走 ollama 时自动切到 11434；显式设过就尊重用户的值。
if [[ "${VLM_BACKEND}" == "ollama" && "${VLM_PORT}" == "8080" ]]; then
    VLM_PORT=11434
fi

# 统一的错误输出/退出函数。
die() {
    printf '错误: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf '警告: %s\n' "$*" >&2
}

# --- 参数合法性检查 ---
if ! [[ "${VLM_PORT}" =~ ^[0-9]+$ ]]; then
    die "VLM_PORT 必须是数字端口，当前值: '${VLM_PORT}'"
fi

# --- 后端分支：ollama（推荐）---------------------------------------------
# ollama 原生提供 OpenAI 兼容端点，且 qwen3vl:8b 正是参考实现用来把筒口
# 3D 误差做到 3.17 mm 的那个模型，所以它是一等公民而非"替代方案"。
if [[ "${VLM_BACKEND}" == "ollama" ]]; then
    command -v ollama >/dev/null 2>&1 || die "找不到 ollama 可执行文件，请先安装：https://ollama.com/download"
    printf '\n=== 启动 ollama（OpenAI 兼容端点）===\n'
    printf '  模型   : %s\n' "${VLM_OLLAMA_MODEL}"
    printf '  监听   : http://%s:%s\n' "${VLM_HOST}" "${VLM_PORT}"
    printf '\n提示：模型不在本地时先执行一次  ollama pull %s\n\n' "${VLM_OLLAMA_MODEL}"
    printf '启动后导出（复制下面这行）：\n'
    printf '  export HYBRID_VLM_URL=http://%s:%s/v1/chat/completions\n' "${VLM_HOST}" "${VLM_PORT}"
    printf '  export HYBRID_VLM_MODEL=%s\n\n' "${VLM_OLLAMA_MODEL}"
    OLLAMA_HOST="${VLM_HOST}:${VLM_PORT}" exec ollama serve
fi

# --- 后端分支：llama.cpp（备用）------------------------------------------
# 定位 llama-server：优先显式指定，其次 PATH。
if [[ -z "${LLAMA_SERVER_BIN}" ]]; then
    LLAMA_SERVER_BIN="$(command -v llama-server || true)"
fi
if [[ -z "${LLAMA_SERVER_BIN}" || ! -x "${LLAMA_SERVER_BIN}" ]]; then
    die "找不到可执行的 llama-server。请安装 llama.cpp（编译产物含 llama-server），或设置 LLAMA_SERVER_BIN=/path/to/llama-server；也可用 VLM_BACKEND=ollama 走 ollama，或参考本文件顶部的 vLLM 替代命令。当前值: '${LLAMA_SERVER_BIN}'"
fi

# 权重文件存在性检查（给出可操作提示）。
if [[ ! -f "${MODEL_PATH}" ]]; then
    die "VLM 主模型文件不存在: ${MODEL_PATH}
      请运行 \`bash robosyn_hybrid/scripts/download_assets.sh\` 下载权重，
      或设置 MODEL_PATH=/path/to/model.gguf（也可设置 ROBOSYN_HYBRID_ASSETS 指向资产目录）。"
fi
if [[ -n "${MMPROJ_PATH}" ]]; then
    if [[ ! -f "${MMPROJ_PATH}" ]]; then
        die "多模态投影文件不存在: ${MMPROJ_PATH}
      请运行 \`bash robosyn_hybrid/scripts/download_assets.sh\` 下载 mmproj，
      或设置 MMPROJ_PATH=/path/to/mmproj.gguf；若模型本身已内置视觉塔，可设 MMPROJ_PATH= 跳过 --mmproj。"
    fi
else
    warn "MMPROJ_PATH 为空：本次启动不加 --mmproj（仅当模型内置视觉塔时才正确）。"
fi

# --- 打印配置摘要与给用户复制的环境变量 ---
printf '启动本地 VLM 服务:\n'
printf '  llama-server : %s\n' "${LLAMA_SERVER_BIN}"
printf '  模型         : %s\n' "${MODEL_PATH}"
printf '  mmproj       : %s\n' "${MMPROJ_PATH:-<未使用>}"
printf '  监听         : http://%s:%s\n' "${VLM_HOST}" "${VLM_PORT}"
printf '  上下文/GPU层 : %s / %s\n' "${VLM_CTX_SIZE}" "${VLM_GPU_LAYERS}"
printf '\n在另一个终端里导出以下变量后即可让 robosyn_hybrid.vlm 使用该服务:\n'
printf '  export HYBRID_VLM_URL=http://%s:%s/v1/chat/completions\n' "${VLM_HOST}" "${VLM_PORT}"
printf '  export HYBRID_VLM_MODEL=%s\n' "${HYBRID_VLM_MODEL:-qwen3-vl}"
printf '\n'

# --- 组装并执行 llama-server 命令 ---
server_args=(
    --model "${MODEL_PATH}"
    --host "${VLM_HOST}"
    --port "${VLM_PORT}"
    --ctx-size "${VLM_CTX_SIZE}"
    --n-gpu-layers "${VLM_GPU_LAYERS}"
)
if [[ -n "${MMPROJ_PATH}" ]]; then
    server_args+=(--mmproj "${MMPROJ_PATH}")
fi
if [[ -n "${VLM_EXTRA_ARGS}" ]]; then
    # 按空格拆分成多个参数（不支持带空格的取值，简单场景足够）。
    # shellcheck disable=SC2206
    extra_args=(${VLM_EXTRA_ARGS})
    server_args+=("${extra_args[@]}")
fi

printf '实际执行: %s %s\n\n' "${LLAMA_SERVER_BIN}" "${server_args[*]}"
exec "${LLAMA_SERVER_BIN}" "${server_args[@]}"
