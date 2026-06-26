#!/usr/bin/env bash
set -euo pipefail

HF_REPO="${1:-YOUR_HF_ORG/EnsembleVLA-ICML2026-checkpoints}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required. Install it with: pip install -U huggingface_hub"
  exit 1
fi

echo "Downloading DP + DP3 base checkpoints from ${HF_REPO}"
echo "Destination: ${ROOT_DIR}"

huggingface-cli download "${HF_REPO}" \
  --local-dir "${ROOT_DIR}" \
  --include "policy/DP/checkpoints/**" \
  --include "policy/DP3/checkpoints/**"

echo "Done. See docs/dp_dp3_checkpoints.md for the expected file layout."
