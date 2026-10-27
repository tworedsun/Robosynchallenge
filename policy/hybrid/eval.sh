#!/bin/bash
# ----------------------------------------------------------------------------
# RoboSynChallenge hybrid policy evaluation.
#
#   bash policy/hybrid/eval.sh <TASK_NAME> <SETTING> [MODEL_NAME] [GPU_ID] [extra overrides...]
#
# Positional arguments:
#   TASK_NAME   one of the ten simulation tasks; handle_basket / click_bell /
#               items_handover use the in-repo hybrid flows, the other seven use
#               the released baseline ACT checkpoints
#   SETTING     random | clear
#   MODEL_NAME  run label (default: hybrid_v1, or $HYBRID_MODEL_NAME)
#   GPU_ID      value exported as CUDA_VISIBLE_DEVICES (default: 0)
#   extras      any additional `--overrides` pairs forwarded to eval_policy.py
#
#   The three-argument form `eval.sh <TASK_NAME> <SETTING> <GPU_ID>` is also
#   accepted: a bare integer in third position with no fourth argument is read
#   as the GPU id.
#
# Environment variables:
#   HYBRID_CKPT        -> --checkpoint_path (local ACT checkpoint directory)
#   HYBRID_ACT_CHUNKS  -> --hybrid_act_chunks
#   HYBRID_ACT_STEP    -> --hybrid_act_step
#   HYBRID_ENABLED     -> --hybrid_enabled (true/false)
#   HYBRID_MODEL_NAME  -> default MODEL_NAME when not given positionally
#   EMBODICHAIN_ROOT   -> extra PYTHONPATH entry (default: <repo>/../EmbodiChain)
#   PYTHON_BIN         -> python interpreter (prefers <this dir>/.venv/bin/python)
#
# No absolute path is hard-coded: every location below is derived from the
# location of this script.
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -z "${PYTHON_BIN:-}" && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

POLICY_NAME=hybrid

TASK_NAME="${1:-handle_basket}"
SETTING="${2:-random}"
ARG3="${3:-}"
ARG4="${4:-}"

# Tolerate the `<task> <setting> <gpu_id> [extras...]` shorthand.
MODEL_NAME=""
GPU_ID=""
SHIFT_COUNT=4
if [[ "$ARG3" =~ ^[0-9]+$ ]] && { [[ -z "$ARG4" ]] || [[ "$ARG4" == -* ]]; }; then
    GPU_ID="$ARG3"
    SHIFT_COUNT=3
else
    MODEL_NAME="$ARG3"
    GPU_ID="$ARG4"
fi
MODEL_NAME="${MODEL_NAME:-${HYBRID_MODEL_NAME:-hybrid_v1}}"
GPU_ID="${GPU_ID:-0}"

if (( SHIFT_COUNT > $# )); then
    SHIFT_COUNT=$#
fi
shift "$SHIFT_COUNT" 2>/dev/null || true
EXTRA_ARGS=("$@")

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "========================================="
echo "  Hybrid Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Model name: $MODEL_NAME"
if [[ -n "${HYBRID_CKPT:-}" ]]; then
    echo "  Checkpoint: $HYBRID_CKPT"
else
    echo "  Checkpoint: <from deploy_policy.yml / Hub>"
fi
echo "  Act chunks: ${HYBRID_ACT_CHUNKS:-<from deploy_policy.yml>}"
echo "  GPU:        $GPU_ID"
echo "  Python:     $PYTHON_BIN"
echo "========================================="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: cannot find Python command: $PYTHON_BIN" >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$REPO_ROOT/policy/hybrid${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$EMBODICHAIN_ROOT" ]]; then
    export PYTHONPATH="$EMBODICHAIN_ROOT:$PYTHONPATH"
fi
cd "$REPO_ROOT"

OVERRIDES=(
    --task_name "$TASK_NAME"
    --setting "$SETTING"
    --model_name "$MODEL_NAME"
)
if [[ -n "${HYBRID_CKPT:-}" ]]; then
    OVERRIDES+=(--checkpoint_path "$HYBRID_CKPT")
fi
if [[ -n "${HYBRID_ACT_CHUNKS:-}" ]]; then
    OVERRIDES+=(--hybrid_act_chunks "$HYBRID_ACT_CHUNKS")
fi
if [[ -n "${HYBRID_ACT_STEP:-}" ]]; then
    OVERRIDES+=(--hybrid_act_step "$HYBRID_ACT_STEP")
fi
if [[ -n "${HYBRID_ENABLED:-}" ]]; then
    OVERRIDES+=(--hybrid_enabled "$HYBRID_ENABLED")
fi
if [[ -n "${HYBRID_CHECKPOINT_REVISION:-}" ]]; then
    OVERRIDES+=(--checkpoint_revision "$HYBRID_CHECKPOINT_REVISION")
fi

PYTHONWARNINGS=ignore::UserWarning \
EMBODICHAIN_SIM_EXIT_PROCESS=0 \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config "policy/$POLICY_NAME/deploy_policy.yml" \
    --overrides \
    "${OVERRIDES[@]}" \
    "${EXTRA_ARGS[@]}"
