#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_fs_env.sh —— 为 FoundationStereo 准备一个独立的轻量 venv 与权重。
#
# 设计约束(与 SPEC §0 一致):
#   * 脚本内**不出现任何绝对路径**;所有位置都由环境变量推导;
#   * venv 目录默认落在资产目录下(相对路径),名字用连字符,避免和参考实现
#     里那个写死的 venv 目录名撞车;
#   * 不修改任何全局解释器,不 pip install 到系统环境。
#
# 用法:
#   bash robosyn_hybrid/scripts/setup_fs_env.sh venv      # 建 venv + 装 requirements
#   bash robosyn_hybrid/scripts/setup_fs_env.sh clone     # 把 FoundationStereo 代码拉到 FS_ROOT
#   bash robosyn_hybrid/scripts/setup_fs_env.sh weights   # 下载 model_best_bp2.pth
#   bash robosyn_hybrid/scripts/setup_fs_env.sh serve     # 前台启动常驻 worker(FS_STEREO_SOCKET)
#   bash robosyn_hybrid/scripts/setup_fs_env.sh check     # 打印解析后的路径与存在性
#   bash robosyn_hybrid/scripts/setup_fs_env.sh all       # clone + venv + weights
#
# 建议的导出(打印在结尾,不自动写进 shell 配置):
#   export ROBOSYN_HYBRID_ASSETS=<资产目录>
#   export FS_ROOT=<FS_ROOT>
#   export FS_CKPT=<FS_ROOT>/pretrained_models/11-33-40/model_best_bp2.pth
#   export FS_PYTHON=<venv>/bin/python
#   export FS_STEREO_SOCKET=<socket 路径>
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# 资产目录:环境变量优先,否则放到提交包旁边的 robosyn_hybrid_assets。
ASSETS_DIR="${ROBOSYN_HYBRID_ASSETS:-${SUBMISSION_ROOT}/robosyn_hybrid_assets}"
FS_ROOT_DIR="${FS_ROOT:-${ASSETS_DIR}/FoundationStereo}"
FS_VENV_DIR="${FS_VENV:-${ASSETS_DIR}/fs-venv}"
FS_CKPT_PATH="${FS_CKPT:-${FS_ROOT_DIR}/pretrained_models/11-33-40/model_best_bp2.pth}"
# socket 留空表示"交给 robosyn_hybrid.settings 决定"(FS_STEREO_SOCKET),
# 本脚本不发明任何默认路径。
FS_SOCKET_PATH="${FS_STEREO_SOCKET:-}"
FS_REPO_URL="${FS_REPO_URL:-https://github.com/NVlabs/FoundationStereo.git}"
FS_REPO_REV="${FS_REPO_REV:-main}"
# 权重来源。官方 README 把 11-33-40 / 23-51-11 放在 Google Drive 目录里;
# 有直链时优先用 FS_CKPT_URL(例如自建镜像),否则用 gdown 拉整个目录。
FS_CKPT_URL="${FS_CKPT_URL:-}"
FS_WEIGHTS_FOLDER_URL="${FS_WEIGHTS_FOLDER_URL:-https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing}"
FS_WEIGHTS_SUBDIR="${FS_WEIGHTS_SUBDIR:-11-33-40}"

SYSTEM_PYTHON="${PYTHON:-python3}"
VENV_PYTHON="${FS_VENV_DIR}/bin/python"
REQUIREMENTS_FILE="${SUBMISSION_ROOT}/robosyn_hybrid/requirements/foundationstereo.txt"
FS_INFER_SCRIPT="${SUBMISSION_ROOT}/robosyn_hybrid/fs_infer.py"

log()  { printf '[setup_fs_env] %s\n' "$*"; }
die()  { printf '[setup_fs_env] ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "缺少命令 '$1'"; }

cmd_check() {
  log "SUBMISSION_ROOT = ${SUBMISSION_ROOT}"
  log "ASSETS_DIR      = ${ASSETS_DIR}   (存在: $([ -d "${ASSETS_DIR}" ] && echo yes || echo no))"
  log "FS_ROOT         = ${FS_ROOT_DIR}   (存在: $([ -d "${FS_ROOT_DIR}" ] && echo yes || echo no))"
  log "FS_CKPT         = ${FS_CKPT_PATH}   (存在: $([ -f "${FS_CKPT_PATH}" ] && echo yes || echo no))"
  log "FS_VENV         = ${FS_VENV_DIR}   (存在: $([ -d "${FS_VENV_DIR}" ] && echo yes || echo no))"
  log "FS_PYTHON       = ${VENV_PYTHON}"
  if [ -n "${FS_SOCKET_PATH}" ]; then
    log "FS_STEREO_SOCKET= ${FS_SOCKET_PATH}"
  else
    log "FS_STEREO_SOCKET= (未设置,由 robosyn_hybrid.settings 的默认值决定)"
  fi
  log "requirements    = ${REQUIREMENTS_FILE}   (存在: $([ -f "${REQUIREMENTS_FILE}" ] && echo yes || echo no))"
  log "fs_infer.py     = ${FS_INFER_SCRIPT}   (存在: $([ -f "${FS_INFER_SCRIPT}" ] && echo yes || echo no))"
  if [ -x "${VENV_PYTHON}" ]; then
    log "venv python     = $("${VENV_PYTHON}" --version 2>&1)"
  fi
  if [ -x "${VENV_PYTHON}" ]; then
    "${VENV_PYTHON}" - <<'PY' || true
try:
    import torch
    print(f"[setup_fs_env] torch {torch.__version__} cuda={torch.cuda.is_available()}")
except Exception as exc:
    print(f"[setup_fs_env] torch 不可用: {type(exc).__name__}: {exc}")
PY
  fi
}

cmd_clone() {
  need git
  mkdir -p -- "${ASSETS_DIR}"
  if [ -d "${FS_ROOT_DIR}/.git" ]; then
    log "FS_ROOT 已存在,跳过 clone: ${FS_ROOT_DIR}"
  elif [ -d "${FS_ROOT_DIR}" ] && [ -n "$(ls -A -- "${FS_ROOT_DIR}" 2>/dev/null || true)" ]; then
    die "FS_ROOT 已存在且非空,但不是 git 仓库: ${FS_ROOT_DIR}(请移走或设置 FS_ROOT)"
  else
    log "clone ${FS_REPO_URL} (${FS_REPO_REV}) -> ${FS_ROOT_DIR}"
    git clone --recurse-submodules "${FS_REPO_URL}" "${FS_ROOT_DIR}"
    if [ "${FS_REPO_REV}" != "main" ]; then
      git -C "${FS_ROOT_DIR}" checkout "${FS_REPO_REV}"
    fi
  fi
}

cmd_venv() {
  need "${SYSTEM_PYTHON}"
  [ -f "${REQUIREMENTS_FILE}" ] || die "找不到依赖清单: ${REQUIREMENTS_FILE}"
  if [ ! -x "${VENV_PYTHON}" ]; then
    log "创建 venv: ${FS_VENV_DIR}"
    "${SYSTEM_PYTHON}" -m venv "${FS_VENV_DIR}"
  else
    log "复用已有 venv: ${FS_VENV_DIR}"
  fi
  "${VENV_PYTHON}" -m pip install --upgrade pip
  log "安装 ${REQUIREMENTS_FILE}"
  # torch 的 CUDA 轮子通常来自 PyTorch 官方索引;CPU 机器可以设置 FS_TORCH_INDEX_URL=""
  FS_TORCH_INDEX_URL="${FS_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
  if [ -n "${FS_TORCH_INDEX_URL}" ]; then
    "${VENV_PYTHON}" -m pip install --extra-index-url "${FS_TORCH_INDEX_URL}" \
      -r "${REQUIREMENTS_FILE}"
  else
    "${VENV_PYTHON}" -m pip install -r "${REQUIREMENTS_FILE}"
  fi
  log "完成。导出 FS_PYTHON=${VENV_PYTHON}"
}

cmd_weights() {
  if [ -f "${FS_CKPT_PATH}" ]; then
    log "checkpoint 已存在: ${FS_CKPT_PATH}"
    return 0
  fi
  mkdir -p -- "$(dirname -- "${FS_CKPT_PATH}")"

  if [ -n "${FS_CKPT_URL}" ]; then
    log "按 FS_CKPT_URL 直接下载 -> ${FS_CKPT_PATH}"
    if command -v curl >/dev/null 2>&1; then
      curl -fL --retry 3 -o "${FS_CKPT_PATH}.part" "${FS_CKPT_URL}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${FS_CKPT_PATH}.part" "${FS_CKPT_URL}"
    else
      die "需要 curl 或 wget 才能直接下载 checkpoint(或手动放到 ${FS_CKPT_PATH})"
    fi
    mv -- "${FS_CKPT_PATH}.part" "${FS_CKPT_PATH}"
    log "checkpoint 就绪: ${FS_CKPT_PATH}"
    return 0
  fi

  # 官方权重托管在 Google Drive 目录里(见 FoundationStereo readme 的
  # "Model Weights" 一节),用 gdown 拉取;失败时给出可执行的下一步。
  local gdown_py="${VENV_PYTHON}"
  [ -x "${gdown_py}" ] || gdown_py="${SYSTEM_PYTHON}"
  if ! "${gdown_py}" -c "import gdown" >/dev/null 2>&1; then
    die "既没有 FS_CKPT_URL,也没有可用的 gdown。请先 'bash robosyn_hybrid/scripts/setup_fs_env.sh venv',或设置 FS_CKPT_URL,或手动把 model_best_bp2.pth(与其 cfg.yaml)放到 ${FS_CKPT_PATH}"
  fi

  local stage="${ASSETS_DIR}/weights-stage"
  mkdir -p -- "${stage}"
  log "gdown 拉取官方权重目录 -> ${stage}(Google Drive,可能受网络限制)"
  "${gdown_py}" -m gdown --folder "${FS_WEIGHTS_FOLDER_URL}" -O "${stage}" --remaining-ok \
    || log "gdown 返回非零,继续在暂存目录里查找已下到的权重"

  local found
  found="$(find "${stage}" -name 'model_best_bp2.pth' -path "*${FS_WEIGHTS_SUBDIR}*" 2>/dev/null | head -n 1)"
  if [ -z "${found}" ]; then
    found="$(find "${stage}" -name 'model_best_bp2.pth' 2>/dev/null | head -n 1)"
  fi
  if [ -z "${found}" ]; then
    die "在 ${stage} 里没找到 model_best_bp2.pth。请设置 FS_CKPT_URL 指向直链,或手动放到 ${FS_CKPT_PATH}(官方权重目录: ${FS_WEIGHTS_FOLDER_URL})"
  fi

  cp -f -- "${found}" "${FS_CKPT_PATH}"
  # build_model() 要求 cfg.yaml 与 checkpoint 同目录,一并搬过去。
  local cfg_src
  cfg_src="$(dirname -- "${found}")/cfg.yaml"
  if [ -f "${cfg_src}" ]; then
    cp -f -- "${cfg_src}" "$(dirname -- "${FS_CKPT_PATH}")/cfg.yaml"
  fi
  log "checkpoint 就绪: ${FS_CKPT_PATH}"
}

cmd_serve() {
  [ -f "${FS_INFER_SCRIPT}" ] || die "找不到 ${FS_INFER_SCRIPT}"
  [ -x "${VENV_PYTHON}" ] || die "venv 不存在,请先运行: bash robosyn_hybrid/scripts/setup_fs_env.sh venv"
  if [ -n "${FS_SOCKET_PATH}" ]; then
    log "启动常驻 worker,socket=${FS_SOCKET_PATH}"
  else
    log "启动常驻 worker,socket 由 settings 默认值决定"
  fi
  export FS_ROOT="${FS_ROOT_DIR}"
  export FS_CKPT="${FS_CKPT_PATH}"
  if [ -n "${FS_SOCKET_PATH}" ]; then
    export FS_STEREO_SOCKET="${FS_SOCKET_PATH}"
  fi
  exec "${VENV_PYTHON}" "${FS_INFER_SCRIPT}" --serve --fs-root "${FS_ROOT_DIR}"
}

cmd_all() { cmd_clone; cmd_venv; cmd_weights; cmd_check; }

main() {
  local action="${1:-check}"
  case "${action}" in
    venv)    cmd_venv ;;
    clone)   cmd_clone ;;
    weights) cmd_weights ;;
    serve)   cmd_serve ;;
    check)   cmd_check ;;
    all)     cmd_all ;;
    -h|--help|help)
      sed -n '2,30p' "${BASH_SOURCE[0]}"
      return 0
      ;;
    *) die "未知子命令 '${action}'(可用: venv|clone|weights|serve|check|all)" ;;
  esac
  log "导出提示:"
  log "  export ROBOSYN_HYBRID_ASSETS=${ASSETS_DIR}"
  log "  export FS_ROOT=${FS_ROOT_DIR}"
  log "  export FS_CKPT=${FS_CKPT_PATH}"
  log "  export FS_PYTHON=${VENV_PYTHON}"
  if [ -n "${FS_SOCKET_PATH}" ]; then
    log "  export FS_STEREO_SOCKET=${FS_SOCKET_PATH}"
  fi
}

main "$@"
