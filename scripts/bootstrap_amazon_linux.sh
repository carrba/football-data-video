#!/usr/bin/env bash

# sudo dnf install git -y
# sudo dnf install python3.11
# ### sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
# /usr/bin/python3.11 -m venv myenv
# source myenv/bin/activate

# pip install --upgrade pip
# mkdir -p ~/pip-tmp
# export TMPDIR=~/pip-tmp
# source ~/.bashrc
# TMPDIR=~/pip-tmp pip install .

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_VERSION_MIN="3.11"
PYTHON_VERSION_TARGET="${PYTHON_VERSION_TARGET:-3.11.12}"

log() {
  printf '[bootstrap] %s\n' "$*" >&2
}

run_as_root() {
  if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

detect_pkg_manager() {
  if command -v dnf >/dev/null 2>&1; then
    printf 'dnf'
  elif command -v yum >/dev/null 2>&1; then
    printf 'yum'
  else
    return 1
  fi
}

python_version_meets_minimum() {
  local python_executable="$1"
  "$python_executable" - <<'PY'
import sys

major, minor = sys.version_info[:2]
raise SystemExit(0 if (major, minor) >= (3, 11) else 1)
PY
}

install_system_packages() {
  local pkg_manager="$1"

  log "Installing system packages with $pkg_manager"
  run_as_root "$pkg_manager" install -y \
    python3.11 \
    python3-pip \
    git \
    patch \
    gcc \
    make \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    zlib-devel \
    readline-devel \
    sqlite-devel \
    xz-devel \
    tk-devel \
    curl \
    tar \
    pciutils \
    libxcb \
    libX11 \
    libXext \
    libSM \
    libXrender \
    mesa-libGL
}

install_ffmpeg_if_available() {
  local pkg_manager="$1"

  if command -v ffmpeg >/dev/null 2>&1; then
    log "ffmpeg already present"
    return
  fi

  # Not in the default Amazon Linux repos on every AMI; best-effort only since
  # annotated-video output still works without it (just not faststart-remuxed).
  if run_as_root "$pkg_manager" install -y ffmpeg; then
    log "Installed ffmpeg"
    return
  fi

  log "ffmpeg not available via $pkg_manager; annotated videos will skip the faststart remux step. Install ffmpeg manually (e.g. via a static build) if browser streaming is needed."
}

install_nvidia_driver_if_gpu() {
  local pkg_manager="$1"

  if ! command -v lspci >/dev/null 2>&1; then
    return
  fi

  if ! lspci | grep -qi 'NVIDIA'; then
    log "No NVIDIA GPU detected; skipping driver installation"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi already present; skipping driver installation"
    return
  fi

  log "NVIDIA GPU detected; attempting driver installation"

  # Package names vary by Amazon Linux release and repository availability.
  if run_as_root "$pkg_manager" install -y nvidia-driver-latest-dkms; then
    log "Installed NVIDIA driver package: nvidia-driver-latest-dkms"
    return
  fi

  if run_as_root "$pkg_manager" install -y nvidia-driver; then
    log "Installed NVIDIA driver package: nvidia-driver"
    return
  fi

  if run_as_root "$pkg_manager" install -y cuda-drivers; then
    log "Installed NVIDIA driver package: cuda-drivers"
    return
  fi

  log "Could not install NVIDIA drivers automatically. Install manually and reboot before training on GPU."
}

ensure_python() {
  local pkg_manager="$1"

  if command -v python3.11 >/dev/null 2>&1 && python_version_meets_minimum python3.11; then
    printf '%s\n' "python3.11"
    return
  fi

  if command -v python3 >/dev/null 2>&1 && python_version_meets_minimum python3; then
    printf '%s\n' "python3"
    return
  fi

  install_system_packages "$pkg_manager"

  if command -v python3.11 >/dev/null 2>&1 && python_version_meets_minimum python3.11; then
    printf '%s\n' "python3.11"
    return
  fi

  if command -v python3 >/dev/null 2>&1 && python_version_meets_minimum python3; then
    printf '%s\n' "python3"
    return
  fi

  log "Package manager did not provide Python 3.11+, falling back to pyenv"

  local pyenv_root="$HOME/.pyenv"
  if [[ ! -d "$pyenv_root" ]]; then
    git clone https://github.com/pyenv/pyenv.git "$pyenv_root"
  fi

  export PYENV_ROOT="$pyenv_root"
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"

  if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION_TARGET"; then
    log "Building Python $PYTHON_VERSION_TARGET with pyenv"
    pyenv install "$PYTHON_VERSION_TARGET"
  fi

  pyenv shell "$PYTHON_VERSION_TARGET"
  printf '%s\n' "$(pyenv which python)"
}

create_or_refresh_venv() {
  local python_executable="$1"
  local venv_python="$VENV_DIR/bin/python"

  if [[ -x "$venv_python" ]]; then
    if "$venv_python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
      log "Existing virtual environment found at $VENV_DIR"
      return
    fi

    log "Existing virtual environment uses an older Python version; recreating it"
    rm -rf "$VENV_DIR"
  fi

  log "Creating virtual environment with $python_executable"
  "$python_executable" -m venv "$VENV_DIR"
}

main() {
  cd "$PROJECT_ROOT"

  local pkg_manager
  pkg_manager="$(detect_pkg_manager)"

  log "Using repository root $PROJECT_ROOT"
  log "Minimum Python version required by pyproject.toml is $PYTHON_VERSION_MIN"

  install_system_packages "$pkg_manager"
  install_ffmpeg_if_available "$pkg_manager"
  install_nvidia_driver_if_gpu "$pkg_manager"

  local python_executable
  python_executable="$(ensure_python "$pkg_manager")"

  create_or_refresh_venv "$python_executable"

  log "Upgrading pip tooling"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

  log "Installing project in editable mode"
  "$VENV_DIR/bin/python" -m pip install -e .

  log "Done"
  printf '\nActivate the environment with:\n'
  printf '  source %s/bin/activate\n' "$VENV_DIR"
  printf '\nUseful commands:\n'
  printf '  football-s3 --help\n'
  printf '  football-possession --help\n'
  printf '  football-clips --help\n'
}

main "$@"