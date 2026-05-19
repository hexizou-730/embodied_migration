#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Embodied/embodied_migration}"
REPO_URL="${REPO_URL:-https://github.com/hexizou-730/embodied_migration.git}"

log() {
  printf '\n[vm-bootstrap] %s\n' "$*"
}

if [[ ! -f /etc/os-release ]]; then
  echo "This script must be run inside Ubuntu." >&2
  exit 1
fi

# shellcheck source=/dev/null
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "This script expects Ubuntu, found ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi

log "Installing base developer tools."
sudo apt update
sudo apt install -y git curl ca-certificates build-essential

if [[ -d "$PROJECT_DIR/.git" ]]; then
  log "Project already exists: $PROJECT_DIR"
  git -C "$PROJECT_DIR" pull --ff-only || true
else
  log "Cloning project into $PROJECT_DIR"
  mkdir -p "$(dirname "$PROJECT_DIR")"
  git clone "$REPO_URL" "$PROJECT_DIR"
fi

log "Installing project Python environment without NVIDIA driver setup."
bash "$PROJECT_DIR/scripts/setup_ubuntu_maniskill.sh" --yes --no-driver

if [[ ! -f "$PROJECT_DIR/.env" && -f "$PROJECT_DIR/.env.example" ]]; then
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
  log "Created .env from .env.example. Add OPENROUTER_API_KEY before LLM runs."
fi

cat <<EOF

[vm-bootstrap] Done.

Next:
  cd "$PROJECT_DIR"
  conda activate em-ms
  python -m compileall -q .

For early development, focus on non-GPU work:
  - maniskill_backend profiles/tasks/migration/evaluation
  - CapabilityCard and FailureReport
  - LMP executor and static feedback
  - LLM prompt/debug/logging loop
EOF
