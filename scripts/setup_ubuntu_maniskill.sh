#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-em-ms}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
INSTALL_DRIVER=1
ASSUME_YES=0
RUN_APT_UPGRADE=0
ALLOW_WSL=0
SKIP_APT=0
DRIVER_MODE="${DRIVER_MODE:-desktop}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_ubuntu_maniskill.sh [options]

Options:
  --yes              Run non-interactively.
  --no-driver        Skip NVIDIA driver installation.
  --allow-wsl        Allow WSL2 early-development setup. Implies no native GPU/GUI guarantee.
  --skip-apt         Skip apt update/install. Useful after root-preinstalled WSL packages.
  --upgrade          Run apt upgrade before setup. Reboot if prompted.
  --env-name NAME    Conda environment name. Default: em-ms.
  --conda-dir PATH   Miniforge install path. Default: ~/miniforge3.
  --driver-mode MODE Driver install mode: desktop or gpgpu. Default: desktop.
  -h, --help         Show this help.

Environment variables:
  ENV_NAME, PYTHON_VERSION, CONDA_DIR, DRIVER_MODE
EOF
}

log() {
  printf '\n[setup] %s\n' "$*"
}

warn() {
  printf '\n[setup:warning] %s\n' "$*" >&2
}

die() {
  printf '\n[setup:error] %s\n' "$*" >&2
  exit 1
}

confirm() {
  local prompt="$1"
  if [[ "$ASSUME_YES" -eq 1 ]]; then
    return 0
  fi
  read -r -p "$prompt [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)
      ASSUME_YES=1
      shift
      ;;
    --no-driver)
      INSTALL_DRIVER=0
      shift
      ;;
    --allow-wsl)
      ALLOW_WSL=1
      shift
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --upgrade)
      RUN_APT_UPGRADE=1
      shift
      ;;
    --env-name)
      ENV_NAME="${2:-}"
      [[ -n "$ENV_NAME" ]] || die "--env-name requires a value"
      shift 2
      ;;
    --conda-dir)
      CONDA_DIR="${2:-}"
      [[ -n "$CONDA_DIR" ]] || die "--conda-dir requires a value"
      shift 2
      ;;
    --driver-mode)
      DRIVER_MODE="${2:-}"
      [[ "$DRIVER_MODE" == "desktop" || "$DRIVER_MODE" == "gpgpu" ]] || die "--driver-mode must be desktop or gpgpu"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ -f /etc/os-release ]] || die "This script must be run on Ubuntu Linux."
# shellcheck source=/dev/null
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "This script expects Ubuntu, found: ${PRETTY_NAME:-unknown}"

case "${VERSION_ID:-}" in
  22.04|24.04)
    ;;
  *)
    warn "This project was planned for Ubuntu 22.04/24.04; current system is ${PRETTY_NAME:-unknown}."
    ;;
esac

IS_WSL=0
if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
  IS_WSL=1
  if [[ "$ALLOW_WSL" -eq 0 ]]; then
    die "WSL detected. Use --allow-wsl --no-driver for early development, or use native Ubuntu for ManiSkill GUI/GPU simulation."
  fi
  INSTALL_DRIVER=0
  warn "WSL detected. Continuing in early-development mode: no NVIDIA driver install, no native Vulkan/GPU simulation guarantee."
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  warn "This script is tuned for Linux x86_64. Current architecture: $(uname -m)"
fi

log "Project root: $PROJECT_ROOT"
log "Ubuntu: ${PRETTY_NAME:-unknown}"
log "Conda env: $ENV_NAME, Python $PYTHON_VERSION"

if [[ "$SKIP_APT" -eq 0 ]]; then
  log "Updating apt package index and installing base tools."
  sudo apt update

  if [[ "$RUN_APT_UPGRADE" -eq 1 ]]; then
    log "Running apt upgrade."
    sudo apt upgrade -y
    if [[ -f /var/run/reboot-required ]]; then
      warn "A reboot is required after apt upgrade. Reboot, then run this script again."
      exit 10
    fi
  fi

  APT_PACKAGES=(
    ca-certificates
    curl
    wget
    git
    build-essential
    ubuntu-drivers-common
    libgl1
    libegl1
    libvulkan1
  )

  if [[ "$IS_WSL" -eq 0 ]]; then
    APT_PACKAGES+=(linux-headers-"$(uname -r)")
  fi

  sudo apt install -y "${APT_PACKAGES[@]}"
else
  log "Skipping apt setup because --skip-apt was provided."
fi

if [[ "$SKIP_APT" -eq 0 ]]; then
  if apt-cache show vulkan-tools >/dev/null 2>&1; then
    sudo apt install -y vulkan-tools
  elif apt-cache show vulkan-utils >/dev/null 2>&1; then
    sudo apt install -y vulkan-utils
  else
    warn "Could not find vulkan-tools/vulkan-utils in apt; skipping vulkaninfo CLI."
  fi
fi

if [[ "$INSTALL_DRIVER" -eq 1 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver already appears to be working."
    nvidia-smi || true
  else
    log "Available NVIDIA driver candidates:"
    if [[ "$DRIVER_MODE" == "gpgpu" ]]; then
      sudo ubuntu-drivers list --gpgpu || true
    else
      sudo ubuntu-drivers list || true
    fi

    if confirm "Install the recommended NVIDIA driver using ubuntu-drivers? A reboot is required afterward."; then
      if [[ "$DRIVER_MODE" == "gpgpu" ]]; then
        sudo ubuntu-drivers install --gpgpu
      else
        sudo ubuntu-drivers install
      fi
      warn "NVIDIA driver installation finished. Reboot Ubuntu before relying on GPU/Vulkan."
    else
      warn "Skipped NVIDIA driver installation by user choice."
    fi
  fi
else
  log "Skipping NVIDIA driver installation."
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
  log "Using existing conda: $CONDA_EXE"
elif [[ -x "$CONDA_DIR/bin/conda" ]]; then
  CONDA_EXE="$CONDA_DIR/bin/conda"
  log "Using existing Miniforge: $CONDA_EXE"
else
  log "Installing Miniforge into $CONDA_DIR."
  TMP_INSTALLER="$(mktemp -t miniforge.XXXXXX.sh)"
  curl -LfsS -o "$TMP_INSTALLER" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$TMP_INSTALLER" -b -u -p "$CONDA_DIR"
  rm -f "$TMP_INSTALLER"
  CONDA_EXE="$CONDA_DIR/bin/conda"
  "$CONDA_EXE" init bash || true
fi

CONDA_BASE="$("$CONDA_EXE" info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

log "Preparing conda environment: $ENV_NAME"
conda config --set channel_priority strict
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Environment exists; ensuring Python $PYTHON_VERSION is installed."
  conda install -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
else
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

log "Installing Python dependencies from project requirement files."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install -r "$PROJECT_ROOT/requirements.txt"
conda run -n "$ENV_NAME" python -m pip install -r "$PROJECT_ROOT/requirements-maniskill.txt"

log "Verifying core Python imports."
conda run -n "$ENV_NAME" python - <<'PY'
import sys
print("python:", sys.version.split()[0])

import numpy
print("numpy:", numpy.__version__)

import openai
print("openai:", openai.__version__)

import torch
print("torch:", torch.__version__)
print("torch cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch cuda device:", torch.cuda.get_device_name(0))

import gymnasium
print("gymnasium:", gymnasium.__version__)

import mani_skill
print("mani_skill:", getattr(mani_skill, "__version__", "unknown"))
PY

log "GPU/Vulkan quick checks."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || warn "nvidia-smi failed. If the driver was just installed, reboot and try again."
else
  warn "nvidia-smi not found."
fi

if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary || warn "vulkaninfo failed. Check NVIDIA driver/Vulkan ICD after reboot."
else
  warn "vulkaninfo not found."
fi

cat <<EOF

[setup] Done.

Next commands:
  conda activate $ENV_NAME
  python -m mani_skill.examples.demo_random_action -e PickCube-v1 --render-mode human

If NVIDIA driver was installed or apt upgraded the kernel, reboot first:
  sudo reboot
EOF
