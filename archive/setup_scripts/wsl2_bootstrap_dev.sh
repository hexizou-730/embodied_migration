#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Embodied/embodied_migration}"
REPO_URL="${REPO_URL:-https://github.com/hexizou-730/embodied_migration.git}"
SKIP_APT=0

for arg in "$@"; do
  case "$arg" in
    --skip-apt)
      SKIP_APT=1
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

log() {
  printf '\n[wsl2-bootstrap] %s\n' "$*"
}

if [[ ! -f /etc/os-release ]]; then
  echo "This script must be run inside Ubuntu on WSL2." >&2
  exit 1
fi

# shellcheck source=/dev/null
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "This script expects Ubuntu, found ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi

if ! grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
  echo "This does not look like WSL. Use setup_ubuntu_maniskill.sh directly on native Ubuntu." >&2
  exit 1
fi

if [[ "$SKIP_APT" -eq 0 ]]; then
  log "Installing base developer tools."
  sudo apt update
  sudo apt install -y git curl ca-certificates build-essential
else
  log "Skipping apt base tools because --skip-apt was provided."
fi

if [[ -d "$PROJECT_DIR/.git" ]]; then
  log "Project already exists: $PROJECT_DIR"
  git -C "$PROJECT_DIR" pull --ff-only || true
else
  log "Cloning project into $PROJECT_DIR"
  mkdir -p "$(dirname "$PROJECT_DIR")"
  git clone "$REPO_URL" "$PROJECT_DIR"
fi

log "Installing project Python environment in WSL2 early-development mode."
SETUP_ARGS=(--yes --no-driver --allow-wsl)
if [[ "$SKIP_APT" -eq 1 ]]; then
  SETUP_ARGS+=(--skip-apt)
fi
bash "$PROJECT_DIR/scripts/setup_ubuntu_maniskill.sh" "${SETUP_ARGS[@]}"

if [[ ! -f "$PROJECT_DIR/.env" && -f "$PROJECT_DIR/.env.example" ]]; then
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
  log "Created .env from .env.example. Add OPENROUTER_API_KEY or DEEPSEEK_API_KEY before LLM runs."
fi

cat <<EOF

[wsl2-bootstrap] Done.

Next:
  cd "$PROJECT_DIR"
  conda activate em-ms
  python -m compileall -q .

Good WSL2 tasks now:
  - Capability Cards and Failure Reports
  - prompt assembly and LLM calls
  - static feedback baseline
  - migration/evaluation/logging code
  - unit tests and result-analysis scripts

Leave final ManiSkill GUI/GPU validation for native Ubuntu.
EOF
