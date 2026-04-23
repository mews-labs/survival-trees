#!/usr/bin/env bash
# Build and install survival-trees with its Rust extension.
set -euo pipefail

# 1. Rust toolchain (stable) — skip if already installed
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain stable --profile minimal
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

# 2. Python build tool
python3 -m pip install --upgrade pip
python3 -m pip install maturin

# 3. Build + install the extension in the current Python environment
maturin develop --release
