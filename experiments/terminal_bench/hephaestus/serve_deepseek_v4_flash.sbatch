#!/bin/bash
#SBATCH --job-name=gv-dsv4f-pilot
#SBATCH --partition=batch
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

SCRATCH_ROOT="${GV_SCRATCH_ROOT:-/scratch/$USER}"
MODEL_REVISION="7872f01b1d1fe23eabc4c98b48bffcef5a386062"
MODEL_CACHE_DIR="models--deepseek-ai--DeepSeek-V4-Flash-0731"
RUNTIME_ARCHIVE="$SCRATCH_ROOT/gv-pydeps-full-vllm-0.27.1.tar"
RUNTIME_CHECKSUM="$RUNTIME_ARCHIVE.sha256"
RUNTIME_OVERLAY="$HOME/gv-jobs/gv-boto3-overlay-vllm-0.27.1.tar"
RUNTIME_OVERLAY_CHECKSUM="$RUNTIME_OVERLAY.sha256"
PYTHON_DEV_ROOT="${GV_PYTHON_DEV_ROOT:-$HOME/gv-runtime/python312-dev-3.12.3/root}"
DEFAULT_HF_HOME="$HOME/.cache/huggingface"
DEFAULT_VLLM_PYTHON="$HOME/gv-vllm/bin/python"
DEFAULT_MODEL_ID="deepseek-ai/DeepSeek-V4-Flash-0731"
if [[ -f "$SCRATCH_ROOT/gv-hf/.deepseek-v4-flash-0731-complete" ]]; then
  DEFAULT_HF_HOME="$SCRATCH_ROOT/gv-hf"
  DEFAULT_MODEL_ID="$SCRATCH_ROOT/gv-hf/hub/$MODEL_CACHE_DIR/snapshots/$MODEL_REVISION"
fi
if [[ -f "$SCRATCH_ROOT/gv-vllm/.stage-complete" \
      && -x "$SCRATCH_ROOT/gv-vllm/bin/python" ]]; then
  DEFAULT_VLLM_PYTHON="$SCRATCH_ROOT/gv-vllm/bin/python"
fi

# Home directories are NFS-mounted on Hephaestus.  vLLM imports thousands of
# small Python modules, so unpack the pinned runtime archive once into the
# node-local filesystem and reuse it across jobs on that node.
if [[ -s "$RUNTIME_ARCHIVE" && -s "$RUNTIME_CHECKSUM" \
      && -s "$RUNTIME_OVERLAY" && -s "$RUNTIME_OVERLAY_CHECKSUM" ]]; then
  sha256sum --check --status "$RUNTIME_CHECKSUM"
  sha256sum --check --status "$RUNTIME_OVERLAY_CHECKSUM"
  BASE_RUNTIME_DIGEST="$(cut -d' ' -f1 "$RUNTIME_CHECKSUM")"
  OVERLAY_RUNTIME_DIGEST="$(cut -d' ' -f1 "$RUNTIME_OVERLAY_CHECKSUM")"
  RUNTIME_DIGEST="$(
    printf '%s:%s' "$BASE_RUNTIME_DIGEST" "$OVERLAY_RUNTIME_DIGEST" \
      | sha256sum | cut -d' ' -f1
  )"
  LOCAL_TMP_ROOT=/tmp
  LOCAL_PYDEPS="$LOCAL_TMP_ROOT/${USER}-gv-pydeps-full"
  LOCAL_VENV="$LOCAL_TMP_ROOT/${USER}-gv-runtime"
  for local_target in "$LOCAL_PYDEPS" "$LOCAL_VENV"; do
    case "$local_target" in
      /tmp/* | /var/tmp/*) ;;
      *)
        echo "unsafe local runtime target: $local_target" >&2
        exit 2
        ;;
    esac
  done
  exec 9>"$LOCAL_PYDEPS.lock"
  flock 9
  if [[ ! -f "$LOCAL_PYDEPS/.archive-sha256" ]] \
      || [[ "$(<"$LOCAL_PYDEPS/.archive-sha256")" != "$RUNTIME_DIGEST" ]]; then
    rm -rf -- "$LOCAL_PYDEPS"
    tar -C "$LOCAL_TMP_ROOT" -xf "$RUNTIME_ARCHIVE"
    tar -C "$LOCAL_PYDEPS" -xf "$RUNTIME_OVERLAY"
    printf '%s\n' "$RUNTIME_DIGEST" > "$LOCAL_PYDEPS/.archive-sha256"
  fi
  if [[ ! -x "$LOCAL_VENV/bin/python" ]] \
      || [[ ! -f "$LOCAL_VENV/.runtime-sha256" ]] \
      || [[ "$(<"$LOCAL_VENV/.runtime-sha256")" != "$RUNTIME_DIGEST" ]]; then
    rm -rf -- "$LOCAL_VENV"
    /usr/bin/python3 -m venv --without-pip "$LOCAL_VENV"
    printf '%s\n' "$RUNTIME_DIGEST" > "$LOCAL_VENV/.runtime-sha256"
  fi
  flock -u 9
  exec 9>&-
  CUTLASS_DSL_PACKAGES="$LOCAL_PYDEPS/nvidia_cutlass_dsl/dsl_packages"
  if [[ ! -d "$CUTLASS_DSL_PACKAGES/cutlass" ]]; then
    echo "missing CUTLASS DSL package tree: $CUTLASS_DSL_PACKAGES" >&2
    exit 2
  fi
  # Target installs do not execute .pth files. vLLM's pinned CUTLASS wheel
  # exposes the `cutlass` namespace through nvidia_cutlass_dsl_packages.pth,
  # so add that declared package tree explicitly.
  export PYTHONPATH="$CUTLASS_DSL_PACKAGES:$LOCAL_PYDEPS"
  export PATH="$LOCAL_VENV/bin:$LOCAL_PYDEPS/bin:$PATH"
  DEFAULT_VLLM_PYTHON="$LOCAL_VENV/bin/python"
fi

# Cluster login profiles may inject an unrelated HF_HOME.  Use the verified
# experiment cache unless this job explicitly sets the task-scoped override.
export HF_HOME="${GV_HF_HOME:-$DEFAULT_HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_XET_HIGH_PERFORMANCE=1
export VLLM_LOGGING_LEVEL=INFO

# Keep all JIT and kernel caches off the NFS-mounted home directory. DeepGEMM
# compiles kernels during the first profile run, and nvcc can otherwise spend
# minutes blocked in nfs_wait before the API ever becomes ready.
LOCAL_CACHE_ROOT="${GV_LOCAL_CACHE_ROOT:-/tmp/${USER}-gv-dsv4f-vllm-0.27.1}"
case "$LOCAL_CACHE_ROOT" in
  /tmp/* | /var/tmp/*) ;;
  *)
    echo "unsafe local cache root: $LOCAL_CACHE_ROOT" >&2
    exit 2
    ;;
esac
export XDG_CACHE_HOME="$LOCAL_CACHE_ROOT/xdg"
export VLLM_CACHE_ROOT="$LOCAL_CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$LOCAL_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_ROOT/torchinductor"
export TORCH_EXTENSIONS_DIR="$LOCAL_CACHE_ROOT/torch-extensions"
export CUDA_CACHE_PATH="$LOCAL_CACHE_ROOT/cuda"
export NUMBA_CACHE_DIR="$LOCAL_CACHE_ROOT/numba"
export FLASHINFER_WORKSPACE_BASE="$LOCAL_CACHE_ROOT/flashinfer-workspace"
export TMPDIR="$LOCAL_CACHE_ROOT/tmp"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" \
  "$NUMBA_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$TMPDIR"

VLLM_PYTHON="${GV_VLLM_PYTHON:-$DEFAULT_VLLM_PYTHON}"
MODEL_ID="${GV_MODEL_ID:-$DEFAULT_MODEL_ID}"
SERVED_MODEL_NAME="${GV_SERVED_MODEL_NAME:-deepseek-v4-flash-0731-h200-high}"
API_KEY_FILE="${GV_VLLM_API_KEY_FILE:-$HOME/.config/gv/vllm_api_key}"
PORT="${GV_VLLM_PORT:-8371}"

if [[ ! -x "$VLLM_PYTHON" ]]; then
  echo "missing vLLM Python: $VLLM_PYTHON" >&2
  exit 2
fi
if ! command -v ninja >/dev/null 2>&1; then
  echo "missing ninja executable in PATH" >&2
  exit 2
fi
if [[ ! -s "$API_KEY_FILE" ]]; then
  echo "missing API key file: $API_KEY_FILE" >&2
  exit 2
fi
if [[ ! -f "$PYTHON_DEV_ROOT/usr/include/python3.12/Python.h" ]]; then
  echo "missing Python development headers: $PYTHON_DEV_ROOT" >&2
  exit 2
fi

# The cluster Python installation omits development headers. Triton compiles
# a small CUDA utility during vLLM startup, so expose the task-local Ubuntu
# 24.04 Python 3.12 headers to gcc without changing the system installation.
export C_INCLUDE_PATH="$PYTHON_DEV_ROOT/usr/include:$PYTHON_DEV_ROOT/usr/include/python3.12${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"

# vLLM reads this without exposing the token in its argv. The service binds
# the compute-node interface so the login node can forward it, hence auth is
# required even though the endpoint is only used inside the cluster/tunnel.
export VLLM_API_KEY
VLLM_API_KEY="$(<"$API_KEY_FILE")"

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} port=$PORT"
echo "hf_home=$HF_HOME vllm_python=$VLLM_PYTHON"
echo "python_dev_root=$PYTHON_DEV_ROOT"
echo "local_cache_root=$LOCAL_CACHE_ROOT"
echo "ninja=$(command -v ninja)"
echo "model_id=$MODEL_ID served_model=$SERVED_MODEL_NAME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$VLLM_PYTHON" -c \
  'import cutlass; from importlib.metadata import version; print("vllm=" + version("vllm") + " cutlass_dsl=" + version("nvidia-cutlass-dsl"))'

# First-pilot configuration: no DSpark yet. Establish ordinary decoding,
# Anthropic Messages compatibility, tool parsing, and structured output before
# adding speculative decoding as a separate performance variable.
exec "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}' \
  --host 0.0.0.0 \
  --port "$PORT" \
  --disable-uvicorn-access-log
