#!/usr/bin/env bash
# Download every third-party weight the hybrid handle_basket policy needs.
#
# Everything lands in $ROBOSYN_HYBRID_ASSETS (default <repo>/robosyn_hybrid_assets)
# and every large file is checked against a recorded SHA-256, so a truncated or
# substituted download fails loudly instead of silently degrading perception.
#
# USAGE
#   bash policy/hybrid/robosyn_hybrid/scripts/download_assets.sh              # SAM2 + FoundationStereo + VLM
#   bash policy/hybrid/robosyn_hybrid/scripts/download_assets.sh --no-vlm     # skip the ~5.5 GB VLM
#   VLM_SIZE=large bash policy/hybrid/robosyn_hybrid/scripts/download_assets.sh   # 27B VLM used in research
#
# ENVIRONMENT
#   ROBOSYN_HYBRID_ASSETS  destination directory
#   HF_ENDPOINT            default https://hf-mirror.com (huggingface.co is
#                          unreachable from some evaluation networks)
#   FS_SKIP_CODE=1         do not clone the FoundationStereo repository
#   VLM_SIZE               "small" (default, Qwen2.5-VL-7B Q4_K_M) or "large"
#                          (Qwen3.8-27B Q4_K_M, the model used in our research)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ASSETS="${ROBOSYN_HYBRID_ASSETS:-$REPO_ROOT/robosyn_hybrid_assets}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
VLM_SIZE="${VLM_SIZE:-small}"
WANT_VLM=1

for arg in "$@"; do
  case "$arg" in
    --no-vlm) WANT_VLM=0 ;;
    --help|-h) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$ASSETS"
echo "== RoboSynChallenge hybrid asset downloader =="
echo "   assets : $ASSETS"
echo "   mirror : $HF_ENDPOINT"
echo

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
have() { command -v "$1" >/dev/null 2>&1; }

# fetch <url> <dest> [sha256]
fetch() {
  local url="$1" dest="$2" want="${3:-}"
  if [[ -s "$dest" ]]; then
    if [[ -n "$want" ]]; then
      local got
      got="$(sha256sum "$dest" | cut -d' ' -f1)"
      if [[ "$got" == "$want" ]]; then
        echo "  = $(basename "$dest") already present and verified"
        return 0
      fi
      echo "  ! $(basename "$dest") exists but its hash differs; re-downloading"
      rm -f "$dest"
    else
      echo "  = $(basename "$dest") already present"
      return 0
    fi
  fi
  echo "  + $(basename "$dest")"
  local tmp="$dest.part"
  if have curl; then
    curl -L --fail --retry 3 --retry-delay 5 -C - -o "$tmp" "$url"
  elif have wget; then
    wget -c -O "$tmp" "$url"
  else
    echo "need curl or wget" >&2
    return 1
  fi
  mv -f "$tmp" "$dest"
  if [[ -n "$want" ]]; then
    local got
    got="$(sha256sum "$dest" | cut -d' ' -f1)"
    if [[ "$got" != "$want" ]]; then
      echo "  SHA-256 mismatch for $(basename "$dest")" >&2
      echo "    expected $want" >&2
      echo "    got      $got" >&2
      return 1
    fi
    echo "    sha256 ok"
  fi
}

# --------------------------------------------------------------------------- #
# 1. SAM2.1 (Ultralytics build, used through `from ultralytics import SAM`)
# --------------------------------------------------------------------------- #
echo "[1/4] SAM2.1 tiny"
SAM2_SHA="3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3"
fetch \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt" \
  "$ASSETS/sam2.1_t.pt" \
  "$SAM2_SHA"

# --------------------------------------------------------------------------- #
# 2. FoundationStereo source
# --------------------------------------------------------------------------- #
echo "[2/4] FoundationStereo source"
FS_ROOT="$ASSETS/FoundationStereo"
if [[ "${FS_SKIP_CODE:-0}" == "1" ]]; then
  echo "  - skipped (FS_SKIP_CODE=1)"
elif [[ -d "$FS_ROOT/.git" ]]; then
  echo "  = already cloned at $FS_ROOT"
elif have git; then
  echo "  + cloning https://github.com/NVlabs/FoundationStereo"
  git clone --depth 1 https://github.com/NVlabs/FoundationStereo "$FS_ROOT"
else
  echo "  ! git not found; clone https://github.com/NVlabs/FoundationStereo manually to $FS_ROOT" >&2
fi

# --------------------------------------------------------------------------- #
# 3. FoundationStereo weights (11-33-40, ViT-small — the variant this policy uses)
# --------------------------------------------------------------------------- #
echo "[3/4] FoundationStereo weights 11-33-40"
FS_CKPT_DIR="$FS_ROOT/pretrained_models/11-33-40"
mkdir -p "$FS_CKPT_DIR"
FS_SHA="8d7850b9dc68d1366722a02a39745704b1db41471211be6abd93ef463727e6be"
fetch \
  "$HF_ENDPOINT/shriarul5273/FoundationStereo_models/resolve/main/pretrained_models/11-33-40/model_best_bp2.pth" \
  "$FS_CKPT_DIR/model_best_bp2.pth" \
  "$FS_SHA"
# cfg.yaml travels with the checkpoint and is read by the loader.
fetch \
  "$HF_ENDPOINT/shriarul5273/FoundationStereo_models/resolve/main/pretrained_models/11-33-40/cfg.yaml" \
  "$FS_CKPT_DIR/cfg.yaml"

# --------------------------------------------------------------------------- #
# 4. Self-hosted VLM (OpenAI-compatible endpoint via llama.cpp)
# --------------------------------------------------------------------------- #
echo "[4/4] VLM weights"
if [[ "$WANT_VLM" == "0" ]]; then
  echo "  - skipped (--no-vlm)"
else
  VLM_DIR="$ASSETS/vlm"
  mkdir -p "$VLM_DIR"
  if [[ "$VLM_SIZE" == "large" ]]; then
    # The exact model used when the analytic flow was researched (17 GB total).
    echo "  i VLM_SIZE=large -> Qwen3.8-27B Q4_K_M (needs ~24 GB VRAM)"
    fetch "https://www.modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF/resolve/master/Qwen3.8-27B-UD-Q4_K_M.gguf" \
      "$VLM_DIR/Qwen3.8-27B-UD-Q4_K_M.gguf"
    fetch "https://www.modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF/resolve/master/mmproj-F16.gguf" \
      "$VLM_DIR/mmproj-F16.gguf"
  else
    # Default: a 7B vision model that fits on the same 24 GB card as the
    # simulator and the ACT policy. The perception tasks here are a bounding
    # box plus a coarse left/right relation, which does not need 27B.
    echo "  i VLM_SIZE=small -> Qwen2.5-VL-7B-Instruct Q4_K_M (~5.5 GB total)"
    fetch "$HF_ENDPOINT/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" \
      "$VLM_DIR/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    fetch "$HF_ENDPOINT/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" \
      "$VLM_DIR/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
  fi
fi

echo
echo "== done =="
cat <<EOF
Assets are in: $ASSETS

Next steps:
  1. export ROBOSYN_HYBRID_ASSETS="$ASSETS"
  2. build the FoundationStereo environment:
       bash "$SCRIPT_DIR/setup_fs_env.sh"
  3. start the VLM endpoint (prints HYBRID_VLM_URL):
       bash "$SCRIPT_DIR/run_vlm_server.sh"
EOF
